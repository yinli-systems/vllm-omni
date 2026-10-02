# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-scheduled sparse attention under Ulysses matches a single-device reference.

The sparse test kernel drops keys by their position in the complete sequence.
Ulysses hands each rank the full sequence for a subset of heads, so the 2-rank
result equals the 1-rank result only when the adapter runs the sparse kernel on
the post-all-to-all layout and restores the original order afterwards.
"""

import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention import layer as layer_mod
from vllm_omni.diffusion.attention.adapter import DENSE_PATH, SPARSE_PATH, record_attention_dispatch
from vllm_omni.diffusion.attention.backends.abstract import AttentionImpl, SparseAdapterCapabilities
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

SEQ_LEN, NUM_HEADS, HEAD_SIZE, NUM_LAYERS, START_STEP = 64, 8, 16, 3, 1
KEY_BLOCK = 8


class _PositionalBlockSparseImpl(AttentionImpl):
    """SDPA over the keys whose block index is even: a sequence-position sparse rule."""

    def __init__(
        self,
        num_heads,
        head_size,
        softmax_scale,
        causal=False,
        num_kv_heads=None,
        prefix="",
        qkv_layout=None,
        backend_kwargs=None,
        **extra_impl_args,
    ):
        self.softmax_scale = softmax_scale

    def forward(self, query, key, value, attn_metadata=None):
        keep = (torch.arange(key.shape[1], device=key.device) // KEY_BLOCK) % 2 == 0
        q, k, v = (t.transpose(1, 2) for t in (query, key[:, keep], value[:, keep]))
        return F.scaled_dot_product_attention(q, k, v, scale=self.softmax_scale).transpose(1, 2)


class _PositionalBlockSparseBackend(SDPABackend):
    @staticmethod
    def get_name():
        return "TEST_POSITIONAL_SPARSE"

    @staticmethod
    def get_impl_cls():
        return _PositionalBlockSparseImpl

    @classmethod
    def get_sparse_adapter_capabilities(cls):
        return SparseAdapterCapabilities(
            platforms=frozenset({current_omni_platform.device_name}),
            dtypes=frozenset({torch.float32}),
            ulysses=True,
        )


def _worker(rank, world_size, port, input_file, output_file):
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size), MASTER_ADDR="localhost", MASTER_PORT=str(port)
    )
    device = torch.device(f"{current_omni_platform.device_type}:{rank}")
    current_omni_platform.set_device(device)
    init_distributed_environment()
    parallel_config = DiffusionParallelConfig(sequence_parallel_size=world_size, ulysses_degree=world_size)
    od_config = OmniDiffusionConfig.from_kwargs(
        model="test_model",
        dtype=torch.float32,
        parallel_config=parallel_config,
        diffusion_attention_config={
            "default": {
                "backend": "TORCH_SDPA",
                "sparse": {"backend": "TEST_POSITIONAL_SPARSE", "start_step": START_STEP},
            },
        },
    )
    initialize_model_parallel(sequence_parallel_size=world_size, ulysses_degree=world_size)
    layer_mod.get_sparse_attn_backend = lambda *_: _PositionalBlockSparseBackend
    try:
        with set_current_diffusion_config(od_config), set_forward_context(omni_diffusion_config=od_config):
            layers = [
                Attention(
                    num_heads=NUM_HEADS,
                    head_size=HEAD_SIZE,
                    softmax_scale=HEAD_SIZE**-0.5,
                    causal=False,
                    prefix=f"blocks.{i}.attn",
                )
                for i in range(NUM_LAYERS)
            ]
        assert all(attention.step_scheduled is not None for attention in layers)
        q, k, v = (t.to(device) for t in torch.load(input_file))
        shard = slice(rank * SEQ_LEN // world_size, (rank + 1) * SEQ_LEN // world_size)
        outputs = {}
        with record_attention_dispatch() as records:
            for step in (0, START_STEP):
                with set_forward_context(omni_diffusion_config=od_config, denoise_step_idx=step):
                    hidden = q[:, shard]
                    for attention in layers:
                        hidden = attention(hidden, k[:, shard], v[:, shard])
                    outputs[step] = hidden.cpu()
        result = {"outputs": outputs, "paths": [(r.step, r.layer, r.path) for r in records]}
        torch.save(result, output_file.format(rank=rank))
    finally:
        destroy_distributed_env()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_ulysses_sparse_steps_match_single_device():
    if current_omni_platform.get_device_count() < 2:
        pytest.skip("requires 2 devices")
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        input_file = os.path.join(tmp, "qkv.pt")
        torch.save(tuple(torch.randn(1, SEQ_LEN, NUM_HEADS, HEAD_SIZE) for _ in range(3)), input_file)
        runs: dict[int, list[dict]] = {}
        for world_size, port in ((1, 29511), (2, 29512)):
            output_file = os.path.join(tmp, f"out_ws{world_size}_rank{{rank}}.pt")
            torch.multiprocessing.spawn(_worker, args=(world_size, port, input_file, output_file), nprocs=world_size)
            runs[world_size] = []
            for rank in range(world_size):
                runs[world_size].append(torch.load(output_file.format(rank=rank)))

    reference = runs[1][0]["outputs"]
    for step in (0, START_STEP):
        sharded = torch.cat([rank_result["outputs"][step] for rank_result in runs[2]], dim=1)
        torch.testing.assert_close(sharded, reference[step], rtol=1e-5, atol=1e-5)
    # The sparse rule changes the result, so the sparse step really ran the sparse kernel.
    assert not torch.allclose(reference[0], reference[START_STEP])
    for rank_result in runs[1] + runs[2]:
        expected = [
            (step, f"blocks.{i}.attn", DENSE_PATH if step < START_STEP else SPARSE_PATH)
            for step in (0, START_STEP)
            for i in range(NUM_LAYERS)
        ]
        assert rank_result["paths"] == expected
