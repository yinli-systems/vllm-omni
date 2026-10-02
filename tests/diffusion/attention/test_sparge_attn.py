# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SPARGE_ATTN backend: metadata handling (CPU, fake kernel) and kernel accuracy (CUDA)."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends import sparge_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.backends.sparge_attn import SpargeAttentionBackend, SpargeAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

HEADS, HEAD_SIZE = 2, 64


def _impl(**options):
    return SpargeAttentionImpl(
        num_heads=HEADS, head_size=HEAD_SIZE, softmax_scale=HEAD_SIZE**-0.5, backend_kwargs=options or None
    )


def _qkv(seq_len, device="cpu", dtype=torch.float32):
    return tuple(torch.randn(1, seq_len, HEADS, HEAD_SIZE, device=device, dtype=dtype) for _ in range(3))


@pytest.fixture
def fake_kernel(monkeypatch):
    calls = []

    def kernel(q, k, v, *, scale, topk, return_sparsity, **kwargs):
        calls.append({"len": q.shape[1], "topk": topk, "layout": kwargs["tensor_layout"]})
        out = q + 1.0
        return (out, 0.75) if return_sparsity else out

    monkeypatch.setattr(sparge_attn, "_sparge_kernel", lambda: kernel)
    return calls


@pytest.mark.cpu
class TestMetadata:
    def test_no_metadata_runs_whole_sequence(self, fake_kernel):
        q, k, v = _qkv(256)
        out = _impl(topk=0.3).forward_cuda(q, k, v, None)
        assert torch.equal(out, q + 1.0)
        assert fake_kernel == [{"len": 256, "topk": 0.3, "layout": "NHD"}]

    def test_packed_padding_runs_real_rows_and_zero_fills_padding(self, fake_kernel):
        q, k, v = _qkv(320)
        cu = torch.tensor([0, 200, 320], dtype=torch.int32)
        metadata = AttentionMetadata(
            packed_padding=PackedPaddingMetadata(q_length=200, kv_length=200, cu_seqlens_q=cu[:2], cu_seqlens_k=cu[:2]),
            extra={"cu_seqlens_q": cu, "valid_kv_length": 200},
        )
        out = _impl().forward_cuda(q, k, v, metadata)
        assert fake_kernel[0]["len"] == 200
        assert torch.equal(out[:, :200], q[:, :200] + 1.0)
        assert torch.count_nonzero(out[:, 200:]) == 0

    def test_valid_prefix_without_packed_padding(self, fake_kernel):
        q, k, v = _qkv(300)
        metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 180, 300]), "valid_kv_length": 180})
        out = _impl().forward_cuda(q, k, v, metadata)
        assert [c["len"] for c in fake_kernel] == [180]
        assert torch.count_nonzero(out[:, 180:]) == 0

    def test_multi_document_runs_one_call_per_document(self, fake_kernel):
        q, k, v = _qkv(400)
        metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 128, 400]), "valid_kv_length": 400})
        out = _impl().forward_cuda(q, k, v, metadata)
        assert [c["len"] for c in fake_kernel] == [128, 272]
        assert torch.equal(out, q + 1.0)

    def test_short_document_raises_instead_of_dense_fallback(self, fake_kernel):
        q, k, v = _qkv(200)
        metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 100, 200]), "valid_kv_length": 200})
        with pytest.raises(ValueError, match="at least 128 rows"):
            _impl().forward_cuda(q, k, v, metadata)

    @pytest.mark.parametrize(
        "metadata,match",
        [
            (AttentionMetadata(attn_mask=torch.ones(1, 256, dtype=torch.bool)), "attn_mask"),
            (AttentionMetadata(full_attn_spans=[[(0, 4)]]), "piecewise"),
            (AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 300])}), "invalid cu_seqlens"),
        ],
    )
    def test_unsupported_metadata_raises(self, fake_kernel, metadata, match):
        with pytest.raises(ValueError, match=match):
            _impl().forward_cuda(*_qkv(256), metadata)

    def test_packed_layout_requires_batch_one(self, fake_kernel):
        q, k, v = (torch.randn(2, 256, HEADS, HEAD_SIZE) for _ in range(3))
        metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 128, 256]), "valid_kv_length": 256})
        with pytest.raises(ValueError, match="batch size 1"):
            _impl().forward_cuda(q, k, v, metadata)

    def test_sparsity_sampling(self, fake_kernel, monkeypatch):
        logged = []
        monkeypatch.setattr(sparge_attn.logger, "info", lambda *args: logged.append(args))
        impl = _impl(log_sparsity_every=2)
        for _ in range(4):
            impl.forward_cuda(*_qkv(256), None)
        assert len(logged) == 2 and logged[-1][2] == pytest.approx(0.75)


@pytest.mark.cpu
class TestOptionsAndCapabilities:
    @pytest.mark.parametrize(
        "options,match",
        [
            ({"topk": 0}, "topk"),
            ({"topk": 1.5}, "topk"),
            ({"topk": True}, "topk"),
            ({"log_sparsity_every": -1}, "log_sparsity_every"),
            ({"block": 64}, "Unknown SPARGE_ATTN options"),
        ],
    )
    def test_invalid_options(self, options, match):
        with pytest.raises(ValueError, match=match):
            _impl(**options)

    def test_rejects_causal_and_gqa(self):
        with pytest.raises(ValueError, match="causal"):
            SpargeAttentionImpl(num_heads=2, head_size=64, softmax_scale=1.0, causal=True)
        with pytest.raises(ValueError, match="num_kv_heads"):
            SpargeAttentionImpl(num_heads=4, head_size=64, softmax_scale=1.0, num_kv_heads=2)

    def test_capabilities_cover_dense_metadata_contracts(self):
        caps = SpargeAttentionBackend.get_sparse_adapter_capabilities()
        assert caps.packed_mask_free and caps.prefix_kv_slicing and caps.multi_doc_packed_varlen and caps.ulysses
        assert not caps.attention_mask and not caps.causal
        assert caps.option_keys == frozenset(sparge_attn._OPTION_DEFAULTS)
        assert SpargeAttentionBackend.get_supported_head_sizes() == [64, 128]


@hardware_test(res={"cuda": "L4"})
def test_kernel_matches_sdpa_when_all_blocks_kept():
    pytest.importorskip("spas_sage_attn")
    torch.manual_seed(0)
    q, k, v = _qkv(1024, device="cuda", dtype=torch.bfloat16)
    reference = F.scaled_dot_product_attention(*(t.transpose(1, 2).float() for t in (q, k, v))).transpose(1, 2)
    dense = _impl(topk=1.0).forward_cuda(q, k, v, None)
    sparse = _impl(topk=0.25).forward_cuda(q, k, v, None)
    assert dense.dtype == torch.bfloat16 and dense.shape == q.shape and torch.isfinite(sparse).all()
    rel = ((dense.float() - reference).norm() / reference.norm()).item()
    assert rel < 0.06, rel  # INT8 QK / FP8 PV quantization error with every block kept
    assert not torch.allclose(dense, sparse)
