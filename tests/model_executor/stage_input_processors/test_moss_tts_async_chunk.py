# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Progressive Local codec chunks preserve frames and agree with graph shapes."""

from collections import defaultdict
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig
from vllm.config import CompilationConfig, ModelConfig, SchedulerConfig, VllmConfig

from vllm_omni.data_entry_keys import OmniPayloadStruct
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder
from vllm_omni.model_executor.stage_input_processors.moss_tts import talker2codec_raw_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Connector:
    config: dict[str, dict[str, object]]


@dataclass
class _TransferManager:
    connector: _Connector
    code_prompt_token_ids: dict[str, list[object]] = field(default_factory=lambda: defaultdict(list))
    request_payload: dict[str, object] = field(default_factory=dict)
    put_req_chunk: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    ramp_chunk_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class _Request:
    external_req_id: str


def _manager(ramp: object = None, *, first: int = 1, adaptive: bool = False) -> _TransferManager:
    extra: dict[str, object] = {"initial_codec_chunk_frames": first, "codec_chunk_frames": 15, "codec_chunk_ramp": ramp}
    if adaptive:
        extra["codec_chunk_adaptive"] = True
    return _TransferManager(_Connector({"extra": extra}))


def _emit(
    manager: _TransferManager, req_id: str, value: int | None, finished: bool = False
) -> OmniPayloadStruct | None:
    output = None if value is None else {"codes": {"audio": torch.full((1, 4), value)}}
    payload = talker2codec_raw_async_chunk(manager, output, _Request(req_id), finished)
    if payload is not None:
        manager.put_req_chunk[req_id] += 1
        manager.ramp_chunk_count[req_id] += 1
    return payload


@pytest.mark.parametrize(
    "ramp,first,expected",
    [
        (None, 1, [1, 15, 15, 9]),
        ([1, 2, 4, 8, 15], 1, [1, 2, 4, 8, 15, 10]),
        ([1, 4, 15], 7, [1, 4, 15, 15, 5]),
        ([4, 8, 15], 1, [4, 8, 15, 13]),
        ([1, 20, 15], 1, [1, 20, 15, 4]),
        ("1,4,15", 1, [1, 4, 15, 15, 5]),
        ([1, 0, 15], 1, [1, 15, 15, 9]),
        ([1], 1, [1, 15, 15, 9]),
        ("invalid", 1, [1, 15, 15, 9]),
    ],
)
def test_order_and_final_flush(ramp: object, first: int, expected: list[int]) -> None:
    manager = _manager(ramp, first=first)
    packets = []
    for index in range(40):
        packet = _emit(manager, "a", index, finished=index == 39)
        if packet is not None:
            packets.append(packet)
    assert [p.meta.codec_chunk_frames for p in packets] == expected
    actual = torch.cat([p.codes.audio.reshape(4, -1).T for p in packets])
    assert torch.equal(actual, torch.arange(40).unsqueeze(1).expand(-1, 4))
    assert all(not bool(p.meta.finished) for p in packets[:-1])
    assert bool(packets[-1].meta.finished)
    assert "a" not in manager.code_prompt_token_ids


def test_requests_progress_independently_and_empty_finish() -> None:
    manager = _manager([1, 2, 4, 8, 15])
    for name in ("a", "b"):
        first = _emit(manager, name, 0)
        assert first is not None
        assert first.meta.codec_chunk_frames == 1
    assert _emit(manager, "a", 1) is None
    final_a = _emit(manager, "a", None, finished=True)
    assert final_a is not None
    assert final_a.meta.codec_chunk_frames == 1
    assert bool(final_a.meta.finished)
    final_b = _emit(manager, "b", None, finished=True)
    assert final_b is not None
    assert final_b.meta.codec_chunk_frames == 0
    assert bool(final_b.meta.finished)
    assert final_b.codes.audio.numel() == 0
    assert not manager.code_prompt_token_ids


@pytest.mark.parametrize(
    "ramp,first,expected,max_step,effective_first",
    [
        (None, 1, [1, 15], 15, 1),
        ([1, 2, 4, 8, 15], 1, [1, 2, 4, 8, 15], 15, 1),
        ([1, 20, 15], 1, [1, 15, 20], 20, 1),
        ([4, 8, 15], 1, [4, 8, 15], 15, 4),
        ([1, 4, 15], 7, [1, 4, 15], 15, 1),
        ([1, 0, 15], 1, [1, 15], 15, 1),
    ],
)
def test_graph_shapes_and_fast_path_follow_sender(
    ramp: object, first: int, expected: list[int], max_step: int, effective_first: int
) -> None:
    connector = _manager(ramp, first=first).connector.config
    connector["extra"]["codec_first_chunk_fast_path"] = 1
    config = Mock(
        spec=VllmConfig,
        model_config=Mock(
            spec=ModelConfig,
            hf_config=PretrainedConfig(model_type="moss_tts_local"),
            async_chunk=True,
            use_v2_model_runner=True,
            enforce_eager=False,
            stage_connector_config=connector,
        ),
        scheduler_config=Mock(spec=SchedulerConfig, max_num_seqs=8),
        compilation_config=Mock(spec=CompilationConfig, cudagraph_capture_sizes=[1, 2, 4, 8]),
    )
    codec = MossTTSCodecDecoder(vllm_config=config)
    assert codec._streaming_graph_frame_sizes == expected
    assert codec._stream_max_step_frames == max_step
    assert codec._first_chunk_fast
    assert codec._initial_stream_chunk_frames == effective_first


# ---------------------------------------------------------------------------
# Adaptive controller
# ---------------------------------------------------------------------------


def test_adaptive_chunk_0_uses_ic_threshold() -> None:
    """Chunk 0 under adaptive uses the same IC/steady threshold as default."""
    manager = _manager(adaptive=True, first=1)
    packet = _emit(manager, "a", 0)
    assert packet is not None
    assert packet.meta.codec_chunk_frames == 1
    assert not bool(packet.meta.finished)


def test_adaptive_finish_flushes_remaining() -> None:
    """Adaptive flushes all remaining frames on finish."""
    manager = _manager(adaptive=True, first=1)
    # Chunk 0: emit 1 frame (IC threshold = 1).
    first = _emit(manager, "a", 0)
    assert first is not None
    assert first.meta.codec_chunk_frames == 1
    # Feed 1 more frame (pending=1, adaptive target>=2, hold — no emit).
    held = _emit(manager, "a", 1)
    assert held is None
    # Finish with 1 pending → flush.
    final = _emit(manager, "a", None, finished=True)
    assert final is not None
    assert bool(final.meta.finished)
    assert final.meta.codec_chunk_frames == 1
    assert "a" not in manager.code_prompt_token_ids


def test_adaptive_empty_finish_returns_sentinel() -> None:
    """Adaptive returns an empty-finished sentinel when no frames remain."""
    manager = _manager(adaptive=True, first=1)
    # Emit chunk 0, then finish with no pending frames.
    first = _emit(manager, "a", 0)
    assert first is not None
    # Flush remaining with finish + no new frames.
    final = _emit(manager, "a", None, finished=True)
    assert final is not None
    assert bool(final.meta.finished)
    assert final.meta.codec_chunk_frames == 0
    assert final.codes.audio.numel() == 0


def test_adaptive_takes_precedence_over_ramp() -> None:
    """When both codec_chunk_ramp and codec_chunk_adaptive are set, adaptive wins."""
    manager = _manager(ramp=[4, 8, 15], adaptive=True, first=1)
    # Chunk 0: adaptive uses IC=1, not ramp's first entry (4).
    packet = _emit(manager, "a", 0)
    assert packet is not None
    assert packet.meta.codec_chunk_frames == 1


def test_adaptive_requests_progress_independently() -> None:
    """Two requests under adaptive do not interfere."""
    manager = _manager(adaptive=True, first=1)
    # Start both requests — chunk 0 emits 1 frame each.
    a0 = _emit(manager, "a", 0)
    b0 = _emit(manager, "b", 0)
    assert a0 is not None and b0 is not None
    assert a0.meta.codec_chunk_frames == 1
    assert b0.meta.codec_chunk_frames == 1
    # Feed 1 more frame to 'a' (held by adaptive target >= 2).
    assert _emit(manager, "a", 1) is None
    # Finish 'a' with 1 pending → flush 1.
    final_a = _emit(manager, "a", None, finished=True)
    assert final_a is not None
    assert bool(final_a.meta.finished)
    assert final_a.meta.codec_chunk_frames == 1
    # 'b' can still be finished independently.
    final_b = _emit(manager, "b", None, finished=True)
    assert final_b is not None
    assert bool(final_b.meta.finished)


def test_adaptive_states_present_during_request() -> None:
    """The adaptive controller is created on chunk 0 and persists until finish."""
    manager = _manager(adaptive=True, first=1)
    _emit(manager, "a", 0)
    assert hasattr(manager, "_adaptive_states")
    assert "a" in manager._adaptive_states
    # Finish cleans up processor-side pending frames.
    _emit(manager, "a", None, finished=True)
    assert "a" not in manager.code_prompt_token_ids
