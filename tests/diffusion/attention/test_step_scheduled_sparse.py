# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-scheduled dense/sparse attention (``AttentionSpec.sparse``)."""

import pytest
import torch

from vllm_omni.diffusion.attention import adapter as adapter_mod
from vllm_omni.diffusion.attention import layer
from vllm_omni.diffusion.attention.adapter import (
    DENSE_PATH,
    SPARSE_PATH,
    record_attention_dispatch,
    validate_sparse_role,
)
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
    SparseAdapterCapabilities,
)
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, SparseAttentionSpec
from vllm_omni.diffusion.forward_context import set_forward_context, set_forward_context_denoise_total_steps
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

DENSE_VALUE = 1.0
SPARSE_VALUE = 2.0


class _ConstantImpl(AttentionImpl):
    value = DENSE_VALUE

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
        self.backend_kwargs = backend_kwargs
        self.calls = 0

    def forward(self, query, key, value, attn_metadata=None):
        self.calls += 1
        return torch.full_like(query, self.value)


class _DenseImpl(_ConstantImpl):
    value = DENSE_VALUE


class _SparseImpl(_ConstantImpl):
    value = SPARSE_VALUE


class _DenseBackend(SDPABackend):
    @staticmethod
    def get_name():
        return "TEST_DENSE"

    @staticmethod
    def get_impl_cls():
        return _DenseImpl


def _caps(**overrides):
    fields = dict(
        platforms=frozenset({current_omni_platform.device_name}),
        dtypes=frozenset({torch.float32, torch.bfloat16}),
        option_keys=frozenset({"sparsity"}),
    )
    fields.update(overrides)
    return SparseAdapterCapabilities(**fields)


def _sparse_backend(caps=None, head_sizes=()):
    capabilities = _caps() if caps is None else caps

    class _SparseBackend(SDPABackend):
        @staticmethod
        def get_name():
            return "TEST_SPARSE"

        @staticmethod
        def get_impl_cls():
            return _SparseImpl

        @staticmethod
        def get_supported_head_sizes():
            return list(head_sizes)

        @classmethod
        def get_sparse_adapter_capabilities(cls):
            return capabilities

    return _SparseBackend


@pytest.fixture
def build_attention(monkeypatch):
    monkeypatch.setattr(layer, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(layer, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    monkeypatch.setattr(layer.Attention, "_get_active_parallel_strategy", lambda self: self._no_parallel_strategy)

    def build(sparse=None, sparse_backend=None, dense_backend=_DenseBackend, prefix="blocks.0.attn", **kwargs):
        spec = AttentionSpec(backend="TORCH_SDPA", sparse=sparse)
        monkeypatch.setattr(layer, "get_attn_backend_for_role", lambda **_: (dense_backend, spec))
        monkeypatch.setattr(layer, "get_sparse_attn_backend", lambda *_: sparse_backend or _sparse_backend())
        return layer.Attention(
            num_heads=2, head_size=8, softmax_scale=8**-0.5, causal=False, prefix=prefix, role="self", **kwargs
        )

    return build


def _qkv(dtype=torch.float32):
    return tuple(torch.randn(1, 4, 2, 8, dtype=dtype) for _ in range(3))


def _run(attention, step, total=None, metadata=None, dtype=torch.float32):
    with set_forward_context(denoise_step_idx=step):
        set_forward_context_denoise_total_steps(total)
        return attention(*_qkv(dtype), metadata)


# ---------------------------------------------------------------- A1: configuration


class TestSparseAttentionSpec:
    def test_mapping_is_coerced_and_options_copied(self):
        options = {"sparsity": 0.8}
        spec = AttentionSpec(
            backend="FLASH_ATTN", sparse={"backend": "TEST_SPARSE", "start_step": 2, "options": options}
        )
        assert isinstance(spec.sparse, SparseAttentionSpec)
        assert spec.sparse.start_step == 2
        assert spec.sparse.options == options and spec.sparse.options is not options

    def test_defaults(self):
        spec = SparseAttentionSpec(backend="TEST_SPARSE")
        assert spec.start_step == 0 and spec.options == {}

    @pytest.mark.parametrize("start_step", [-1, 1.5, "2", True, None])
    def test_invalid_start_step(self, start_step):
        with pytest.raises(ValueError, match="start_step"):
            SparseAttentionSpec(backend="TEST_SPARSE", start_step=start_step)

    @pytest.mark.parametrize("backend", ["", "  ", 3])
    def test_invalid_backend(self, backend):
        with pytest.raises(TypeError, match="sparse.backend"):
            SparseAttentionSpec(backend=backend)

    def test_auto_backend_rejected(self):
        with pytest.raises(ValueError, match="explicitly"):
            SparseAttentionSpec(backend="auto")

    @pytest.mark.parametrize("options", [[("sparsity", 0.8)], {1: 0.8}])
    def test_invalid_options(self, options):
        with pytest.raises(TypeError, match="sparse.options"):
            SparseAttentionSpec(backend="TEST_SPARSE", options=options)

    def test_rejected_with_self_scheduling_block_sparse_backend(self):
        with pytest.raises(ValueError, match="schedules its own sparse steps"):
            AttentionSpec(backend="RAINFUSION_ATTN", sparse={"backend": "TEST_SPARSE"})

    def test_dense_backend_kwargs_unchanged(self):
        assert AttentionSpec(backend="FLASH_ATTN", sparse={"backend": "TEST_SPARSE"}).backend_kwargs() is None

    @pytest.mark.parametrize("start_step,step,expected", [(0, 0, True), (2, 1, False), (2, 2, True), (2, 3, True)])
    def test_boundary(self, start_step, step, expected):
        assert SparseAttentionSpec(backend="TEST_SPARSE", start_step=start_step).use_sparse(step) is expected

    def test_per_role_entry_accepts_sparse(self):
        config = AttentionConfig(
            per_role={
                "minimax_h3": {"self": {"backend": "FLASH_ATTN", "sparse": {"backend": "TEST_SPARSE", "start_step": 3}}}
            }
        )
        spec, _ = config.resolve_with_source(role="minimax_h3.self")
        assert spec.backend == "FLASH_ATTN" and spec.sparse.start_step == 3
        # Roles without an entry (e.g. the token refiner) keep their dense default.
        assert config.resolve_with_source(role="minimax_h3.token_refiner") == (None, None)


# ---------------------------------------------------------------- A2: contract validation


def _validate(sparse_backend, dense_backend=_DenseBackend, spec=None, **overrides):
    kwargs = dict(
        spec=spec or SparseAttentionSpec(backend="TEST_SPARSE"),
        dense_backend=dense_backend,
        sparse_backend=sparse_backend,
        head_size=8,
        causal=False,
        qkv_layout=None,
        ulysses_degree=1,
        ring_degree=1,
        allgather_degree=1,
        skip_sequence_parallel=False,
    )
    kwargs.update(overrides)
    return validate_sparse_role(**kwargs)


class TestContractValidation:
    def test_backend_without_contract_rejected(self):
        with pytest.raises(ValueError, match="does not implement the step-scheduled sparse contract"):
            _validate(SDPABackend)

    def test_platform_rejected(self):
        with pytest.raises(ValueError, match="does not run on platform"):
            _validate(_sparse_backend(_caps(platforms=frozenset({"no-such-platform"}))))

    def test_head_size_rejected(self):
        with pytest.raises(ValueError, match="head_size=8"):
            _validate(_sparse_backend(head_sizes=(64, 128)))

    def test_causal_rejected(self):
        with pytest.raises(ValueError, match="causal"):
            _validate(_sparse_backend(), causal=True)

    def test_qkv_layout_rejected(self):
        with pytest.raises(ValueError, match="qkv_layout"):
            _validate(_sparse_backend(_caps(qkv_layouts=frozenset({"BHSD"}))), qkv_layout="BSND")

    def test_unknown_option_rejected(self):
        with pytest.raises(ValueError, match="Unknown sparse.options"):
            _validate(_sparse_backend(), spec=SparseAttentionSpec(backend="TEST_SPARSE", options={"topk": 3}))

    @pytest.mark.parametrize(
        "dense_capability,match",
        [
            ("supports_packed_mask_free", "packed mask-free"),
            ("supports_prefix_kv_slicing", "K/V prefix"),
            ("supports_multi_doc_packed_varlen", "isolate packed documents"),
        ],
    )
    def test_metadata_contract_of_dense_backend_must_be_consumable(self, dense_capability, match):
        attrs: dict[str, object] = (
            {dense_capability: True}
            if dense_capability == "supports_prefix_kv_slicing"
            else {dense_capability: classmethod(lambda cls: True)}
        )
        dense = type("_CapableDense", (_DenseBackend,), attrs)
        with pytest.raises(ValueError, match=match):
            _validate(_sparse_backend(), dense_backend=dense)
        capability = {
            "supports_packed_mask_free": "packed_mask_free",
            "supports_prefix_kv_slicing": "prefix_kv_slicing",
            "supports_multi_doc_packed_varlen": "multi_doc_packed_varlen",
        }[dense_capability]
        _validate(_sparse_backend(_caps(**{capability: True})), dense_backend=dense)

    def test_ulysses_requires_capability(self):
        with pytest.raises(ValueError, match="Ulysses"):
            _validate(_sparse_backend(), ulysses_degree=2)
        _validate(_sparse_backend(_caps(ulysses=True)), ulysses_degree=2)
        # Layers outside the sequence-parallel region never see Ulysses tensors.
        _validate(_sparse_backend(), ulysses_degree=2, skip_sequence_parallel=True)

    @pytest.mark.parametrize("degrees,match", [({"ring_degree": 2}, "ring"), ({"allgather_degree": 2}, "AllGather")])
    def test_unsupported_parallelism_rejected(self, degrees, match):
        with pytest.raises(ValueError, match=match):
            _validate(_sparse_backend(_caps(ulysses=True)), **degrees)


# ---------------------------------------------------------------- A3: layer routing


class TestLayerRouting:
    def test_no_sparse_spec_keeps_dense_layer_unchanged(self, build_attention):
        attention = build_attention()
        assert attention.step_scheduled is None
        assert torch.all(_run(attention, step=5) == DENSE_VALUE)

    @pytest.mark.parametrize("start_step", [0, 2, 3])
    def test_routes_by_step(self, build_attention, start_step):
        attention = build_attention(sparse={"backend": "TEST_SPARSE", "start_step": start_step})
        with record_attention_dispatch() as records:
            outputs = [_run(attention, step=step, total=5) for step in range(5)]
        for step, out in enumerate(outputs):
            assert torch.all(out == (SPARSE_VALUE if step >= start_step else DENSE_VALUE))
        assert [r.path for r in records] == [SPARSE_PATH if s >= start_step else DENSE_PATH for s in range(5)]
        assert [r.backend for r in records] == ["TEST_SPARSE" if s >= start_step else "TEST_DENSE" for s in range(5)]
        assert attention.attention.calls == start_step
        assert attention.step_scheduled.sparse_impl.calls == 5 - start_step

    def test_sparse_options_reach_sparse_impl_only(self, build_attention):
        attention = build_attention(sparse={"backend": "TEST_SPARSE", "options": {"sparsity": 0.9}})
        assert attention.step_scheduled.sparse_impl.backend_kwargs == {"sparsity": 0.9}
        assert attention.attention.backend_kwargs is None

    def test_start_step_beyond_total_runs_dense_and_warns(self, build_attention, monkeypatch):
        warnings = []
        monkeypatch.setattr(adapter_mod.logger, "warning", lambda *args: warnings.append(args))
        attention = build_attention(sparse={"backend": "TEST_SPARSE", "start_step": 10})
        for step in range(4):
            assert torch.all(_run(attention, step=step, total=4) == DENSE_VALUE)
        assert len(warnings) == 1

    def test_missing_step_index_raises(self, build_attention):
        attention = build_attention(sparse={"backend": "TEST_SPARSE"})
        with pytest.raises(RuntimeError, match="denoise_step_idx"):
            _run(attention, step=None)

    def test_all_layers_follow_one_policy_per_step(self, build_attention):
        layers = [
            build_attention(sparse={"backend": "TEST_SPARSE", "start_step": 2}, prefix=f"blocks.{i}.attn")
            for i in range(4)
        ]
        with record_attention_dispatch() as records:
            for step in range(4):
                for attention in layers:
                    _run(attention, step=step, total=4)
        by_step: dict[int, set[str]] = {}
        for record in records:
            by_step.setdefault(record.step, set()).add(record.path)
        assert by_step == {0: {DENSE_PATH}, 1: {DENSE_PATH}, 2: {SPARSE_PATH}, 3: {SPARSE_PATH}}
        assert {r.layer for r in records if r.step == 3} == {f"blocks.{i}.attn" for i in range(4)}

    def test_unsupported_mask_raises_instead_of_dense_fallback(self, build_attention):
        attention = build_attention(sparse={"backend": "TEST_SPARSE"})
        metadata = AttentionMetadata(attn_mask=torch.ones(1, 4, dtype=torch.bool))
        attention._assert_metadata_compatible = lambda _: None
        with pytest.raises(ValueError, match="cannot apply attn_mask"):
            _run(attention, step=0, metadata=metadata)
        assert attention.attention.calls == 0

    def test_unsupported_dtype_raises(self, build_attention):
        attention = build_attention(
            sparse={"backend": "TEST_SPARSE"}, sparse_backend=_sparse_backend(_caps(dtypes=frozenset({torch.bfloat16})))
        )
        with pytest.raises(ValueError, match="dtype"):
            _run(attention, step=0, dtype=torch.float32)

    def test_compilation_rejected(self, build_attention, monkeypatch):
        attention = build_attention(sparse={"backend": "TEST_SPARSE"})
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        with set_forward_context(denoise_step_idx=0), pytest.raises(NotImplementedError, match="eager"):
            attention._run_local_attention(*_qkv(), None)

    def test_kv_cache_quantization_rejected(self, build_attention, monkeypatch):
        def init_quant(self, config):
            self._kv_cache_dtype = "fp8"

        monkeypatch.setattr(layer.Attention, "_init_kv_cache_quantization", init_quant)
        with pytest.raises(ValueError, match="KV-cache quantization"):
            build_attention(sparse={"backend": "TEST_SPARSE"})

    def test_invalid_contract_fails_at_layer_build(self, build_attention):
        with pytest.raises(ValueError, match="does not implement the step-scheduled sparse contract"):
            build_attention(sparse={"backend": "TEST_SPARSE"}, sparse_backend=SDPABackend)
