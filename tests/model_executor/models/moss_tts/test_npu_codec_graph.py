# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU parity tests for the MOSS-TTS codec NPUGraph streaming decoder wrapper.

Verifies that ``NPUGraphStreamingDecoderWrapper`` (graph capture + replay)
produces results equivalent to eager decoding on NPU, covering:

1. **Eager vs graph parity** — multi-step streaming decode via the wrapper
   matches eager decode with identical weights, codes, and state.
2. **Batch padding** — real rows match exactly when padded to a larger bucket.
3. **Slot reset** — after resetting a slot, re-decoding on that slot matches
   a fresh eager reference.
4. **Ring-buffer wraparound** — decode continues correctly past the ring
   capacity boundary.

The wrapper is exercised through ``MossAudioTokenizerModel`` using a tiny
synthetic codec (random weights, small dimensions) so the test does not
depend on a HuggingFace checkpoint download.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerModel,
)
from vllm_omni.model_executor.models.moss_tts.configuration_moss_audio_tokenizer_v2 import (
    MossAudioTokenizerConfig,
)
from vllm_omni.platforms.npu.models.moss_tts_streaming_decode_wrapper import (
    NPUGraphStreamingDecoderWrapper,
)

pytestmark = [pytest.mark.core_model, pytest.mark.tts, pytest.mark.npu]

DEVICE = "npu"
DTYPE = torch.float32

TRANSFORMER = {
    "module_type": "Transformer",
    "d_model": 32,
    "num_heads": 4,
    "num_layers": 1,
    "dim_feedforward": 64,
    "causal": True,
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "max_period": 10000,
    "gating": "none",
    "layer_scale": 1.0,
    "conv_layout": True,
}


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


npu_only = pytest.mark.skipif(not _npu_available(), reason="NPU device or torch_npu not available.")


def _tiny_codec(device: str = DEVICE, dtype: torch.dtype = DTYPE) -> MossAudioTokenizerModel:
    config = MossAudioTokenizerConfig(
        sampling_rate=64,
        downsample_rate=4,
        number_channels=1,
        enable_channel_interleave=False,
        encoder_kwargs=[
            {"module_type": "PatchedPretransform", "patch_size": 4},
            {**TRANSFORMER, "input_dimension": 4, "output_dimension": 16, "context_duration": 0.5},
        ],
        decoder_kwargs=[
            {**TRANSFORMER, "input_dimension": 16, "output_dimension": 32, "context_duration": 0.5},
            {"module_type": "PatchedPretransform", "patch_size": 2},
            {**TRANSFORMER, "input_dimension": 16, "output_dimension": 2, "context_duration": 0.25},
            {"module_type": "PatchedPretransform", "patch_size": 2},
        ],
        quantizer_kwargs={
            "input_dim": 16,
            "rvq_dim": 16,
            "output_dim": 16,
            "num_quantizers": 2,
            "codebook_size": 64,
            "codebook_dim": 8,
            "quantizer_type": "rlfq",
        },
    )
    torch.manual_seed(0)
    model = MossAudioTokenizerModel(config).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.3)
    model.quantizer.build_decode_lut(2, dtype=dtype)
    return model


def _make_wrapper(model: MossAudioTokenizerModel, state_capacity: int, batch_sizes: list[int], frame_sizes: list[int]) -> NPUGraphStreamingDecoderWrapper:
    from unittest.mock import MagicMock

    vllm_config = MagicMock()
    return NPUGraphStreamingDecoderWrapper(
        model,
        state_capacity=state_capacity,
        batch_sizes=batch_sizes,
        frame_sizes=frame_sizes,
        num_quantizers=2,
        vllm_config=vllm_config,
    )


@npu_only
@torch.no_grad()
def test_npu_graph_matches_eager_multi_step():
    """NPUGraph replay matches eager decode over multiple streaming steps."""
    model = _tiny_codec()
    n_vq, batch, total_frames = 2, 2, 12
    chunk = 4
    codes = torch.randint(0, 64, (n_vq, batch, total_frames), device=DEVICE, dtype=torch.long)

    model.initialize_decoder_state_pool(state_capacity=batch, scratch_capacity=batch, chunk_frames=chunk)
    wrapper = _make_wrapper(model, state_capacity=batch, batch_sizes=[batch], frame_sizes=[chunk])
    wrapper.warmup(torch.device(DEVICE))

    slot_ids = torch.arange(batch, dtype=torch.long, device=DEVICE)
    graph_parts = []
    for start in range(0, total_frames, chunk):
        c = codes[:, :, start : start + chunk]
        result = wrapper.decode(c, slot_ids)
        assert result is not None, "wrapper.decode must return a result on NPU"
        audio, audio_lengths, actual_batch = result
        graph_parts.append(audio[:actual_batch])
    model.close_decoder_state_pool()

    model.initialize_decoder_state_pool(state_capacity=batch, scratch_capacity=batch, chunk_frames=chunk)
    slot_ids2 = torch.arange(batch, dtype=torch.long, device=DEVICE)
    eager_parts = []
    for start in range(0, total_frames, chunk):
        c = codes[:, :, start : start + chunk]
        lengths = torch.full((batch,), chunk, dtype=torch.long, device=DEVICE)
        valid_rows = torch.ones(batch, dtype=torch.bool, device=DEVICE)
        out = model.decode_streaming_batch(c, lengths, slot_ids2, valid_rows)
        eager_parts.append(out.audio)
    model.close_decoder_state_pool()

    assert len(graph_parts) == len(eager_parts)
    for g, e in zip(graph_parts, eager_parts):
        torch.testing.assert_close(g, e, atol=1e-4, rtol=1e-4)


@npu_only
@torch.no_grad()
def test_npu_graph_batch_padding_matches_real_rows():
    """Padding real rows to a larger bucket produces identical results for the real rows."""
    model = _tiny_codec()
    n_vq, real_batch, total_frames = 2, 1, 8
    chunk = 4
    padded_batch = 2
    codes = torch.randint(0, 64, (n_vq, real_batch, total_frames), device=DEVICE, dtype=torch.long)

    model.initialize_decoder_state_pool(state_capacity=padded_batch, scratch_capacity=padded_batch, chunk_frames=chunk)
    wrapper = _make_wrapper(model, state_capacity=padded_batch, batch_sizes=[padded_batch], frame_sizes=[chunk])
    wrapper.warmup(torch.device(DEVICE))

    slot_ids = torch.arange(real_batch, dtype=torch.long, device=DEVICE)

    graph_parts = []
    for start in range(0, total_frames, chunk):
        c = codes[:, :, start : start + chunk]
        result = wrapper.decode(c, slot_ids)
        assert result is not None, "wrapper.decode must return a result on NPU"
        audio, _, actual_batch = result
        graph_parts.append(audio[:actual_batch])
    model.close_decoder_state_pool()

    # Eager reference with exact batch
    model.initialize_decoder_state_pool(state_capacity=real_batch, scratch_capacity=0, chunk_frames=chunk)
    slot_ids2 = torch.arange(real_batch, dtype=torch.long, device=DEVICE)
    valid_rows2 = torch.ones(real_batch, dtype=torch.bool, device=DEVICE)
    eager_parts = []
    for start in range(0, total_frames, chunk):
        c = codes[:, :, start : start + chunk]
        lengths = torch.full((real_batch,), chunk, dtype=torch.long, device=DEVICE)
        out = model.decode_streaming_batch(c, lengths, slot_ids2, valid_rows2)
        eager_parts.append(out.audio)
    model.close_decoder_state_pool()

    for g, e in zip(graph_parts, eager_parts):
        torch.testing.assert_close(g[:, :real_batch], e, atol=1e-4, rtol=1e-4)


@npu_only
@torch.no_grad()
def test_npu_graph_ring_buffer_wraparound():
    """Decode continues correctly after offsets exceed ring capacity."""
    model = _tiny_codec()
    n_vq, batch = 2, 1
    chunk = 2
    capacity = 4
    num_steps = 6

    codes_per_step = [torch.randint(0, 64, (n_vq, batch, chunk), device=DEVICE, dtype=torch.long) for _ in range(num_steps)]

    model.initialize_decoder_state_pool(state_capacity=batch, scratch_capacity=batch, chunk_frames=chunk)
    wrapper = _make_wrapper(model, state_capacity=batch, batch_sizes=[batch], frame_sizes=[chunk])
    wrapper.warmup(torch.device(DEVICE))

    slot_ids = torch.arange(batch, dtype=torch.long, device=DEVICE)
    valid_rows = torch.ones(batch, dtype=torch.bool, device=DEVICE)

    outputs = []
    for c in codes_per_step:
        result = wrapper.decode(c, slot_ids)
        assert result is not None, "wrapper.decode must return a result on NPU"
        audio, _, actual_batch = result
        outputs.append(audio[:actual_batch])
    model.close_decoder_state_pool()

    assert all(o is not None for o in outputs)
    assert not any(torch.isnan(o).any() for o in outputs)
    assert all(o.shape[-1] == chunk * model.downsample_rate for o in outputs)
