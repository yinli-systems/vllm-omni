# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-scheduled dense/sparse attention for one attention role.

``AttentionSpec.sparse`` keeps the role's dense backend for denoise steps before
``start_step`` and runs the configured sparse backend in every layer of the role
from ``start_step`` on. The decision depends only on the forward context's
``denoise_step_idx``, so every layer and every rank makes the same choice within
a step.

The adapter never falls back to dense attention during a sparse step. Anything
the sparse backend cannot execute exactly is rejected while the layer is built,
or raises when a forward presents it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    SparseAdapterCapabilities,
)
from vllm_omni.diffusion.data import SparseAttentionSpec
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

DENSE_PATH = "dense"
SPARSE_PATH = "sparse"


@dataclass(frozen=True, slots=True)
class AttentionDispatchRecord:
    """One attention call of a step-scheduled role, as actually dispatched."""

    step: int
    layer: str
    role: str
    path: str
    backend: str
    rank: int


_dispatch_records: list[AttentionDispatchRecord] | None = None


@contextlib.contextmanager
def record_attention_dispatch() -> Iterator[list[AttentionDispatchRecord]]:
    """Collect a record for every step-scheduled attention call in this process.

    Recording is off unless this context is active, so the serving path only
    pays a ``None`` check per call.
    """
    global _dispatch_records
    previous = _dispatch_records
    records: list[AttentionDispatchRecord] = []
    _dispatch_records = records
    try:
        yield records
    finally:
        _dispatch_records = previous


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def validate_sparse_role(
    *,
    spec: SparseAttentionSpec,
    dense_backend: type[AttentionBackend],
    sparse_backend: type[AttentionBackend],
    head_size: int,
    causal: bool,
    qkv_layout: str | None,
    ulysses_degree: int,
    ring_degree: int,
    allgather_degree: int,
    skip_sequence_parallel: bool,
) -> SparseAdapterCapabilities:
    """Reject sparse configurations whose sparse steps could not run exactly."""
    name = sparse_backend.get_name()
    caps = sparse_backend.get_sparse_adapter_capabilities()
    if caps is None:
        raise ValueError(f"Attention backend {name!r} does not implement the step-scheduled sparse contract.")

    platform = current_omni_platform.device_name
    if platform not in caps.platforms:
        raise ValueError(f"Sparse backend {name!r} does not run on platform {platform!r}.")
    if not sparse_backend.supports_head_size(head_size):
        raise ValueError(f"Sparse backend {name!r} does not support head_size={head_size}.")
    if causal and not caps.causal:
        raise ValueError(f"Sparse backend {name!r} does not support causal attention.")
    if caps.qkv_layouts is not None and qkv_layout not in caps.qkv_layouts:
        raise ValueError(f"Sparse backend {name!r} does not accept qkv_layout={qkv_layout!r}.")

    unknown = set(spec.options) - caps.option_keys
    if unknown:
        raise ValueError(
            f"Unknown sparse.options for {name!r}: {sorted(unknown)}. Supported options: {sorted(caps.option_keys)}."
        )

    # Models build one metadata object per forward from the dense backend's
    # capabilities. Sparse steps receive the same object, so the sparse kernel
    # must consume each mask-free layout the dense backend lets a model emit.
    dense_name = dense_backend.get_name()
    if dense_backend.supports_packed_mask_free() and not caps.packed_mask_free:
        raise ValueError(
            f"Dense backend {dense_name!r} lets models omit padding masks for packed layouts, "
            f"but sparse backend {name!r} cannot consume packed mask-free metadata."
        )
    if dense_backend.supports_prefix_kv_slicing and not caps.prefix_kv_slicing:
        raise ValueError(
            f"Dense backend {dense_name!r} lets models describe padding as a valid K/V prefix, "
            f"but sparse backend {name!r} cannot slice a K/V prefix."
        )
    if dense_backend.supports_multi_doc_packed_varlen() and not caps.multi_doc_packed_varlen:
        raise ValueError(
            f"Dense backend {dense_name!r} lets models pack several requests into one sequence, "
            f"but sparse backend {name!r} cannot isolate packed documents."
        )

    if not skip_sequence_parallel:
        if ring_degree > 1:
            raise ValueError("Step-scheduled sparse attention does not support ring sequence parallelism.")
        if allgather_degree > 1:
            raise ValueError("Step-scheduled sparse attention does not support AllGather-KV sequence parallelism.")
        if ulysses_degree > 1 and not caps.ulysses:
            raise ValueError(f"Sparse backend {name!r} does not support Ulysses sequence parallelism.")
    return caps


class StepScheduledAttention:
    """Chooses the dense or sparse path of one attention layer for each denoise step."""

    def __init__(
        self,
        *,
        spec: SparseAttentionSpec,
        sparse_backend: type[AttentionBackend],
        capabilities: SparseAdapterCapabilities,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool,
        num_kv_heads: int | None,
        qkv_layout: str | None,
        prefix: str,
        role: str,
        dense_backend_name: str,
    ) -> None:
        self.spec = spec
        self.capabilities = capabilities
        self.sparse_backend_name = sparse_backend.get_name()
        self.dense_backend_name = dense_backend_name
        self.layer = prefix
        self.role = role
        self.sparse_impl: AttentionImpl = sparse_backend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
            prefix=prefix,
            backend_kwargs=dict(spec.options) or None,
            role=role,
            backend_explicit=True,
        )
        self._warned_dense_only = False

    def _current_step(self) -> int:
        context = get_forward_context() if is_forward_context_available() else None
        if context is None or context.denoise_step_idx is None:
            raise RuntimeError(
                f"Attention role {self.role!r} is configured with sparse attention, but the forward context "
                "has no denoise_step_idx. The pipeline must publish its denoise step."
            )
        step = context.denoise_step_idx
        total = context.total_denoise_steps
        if total is not None and self.spec.start_step >= total and not self._warned_dense_only:
            self._warned_dense_only = True
            logger.warning(
                "sparse.start_step=%d is not below the %d denoise steps; role %r runs dense for the whole request.",
                self.spec.start_step,
                total,
                self.role,
            )
        return step

    def select_sparse(self) -> bool:
        """Return whether this call runs the sparse path, recording the decision."""
        if torch.compiler.is_compiling():
            raise NotImplementedError(
                "Step-scheduled sparse attention runs in eager mode only; disable compilation (--enforce-eager)."
            )
        step = self._current_step()
        sparse = self.spec.use_sparse(step)
        if _dispatch_records is not None:
            _dispatch_records.append(
                AttentionDispatchRecord(
                    step=step,
                    layer=self.layer,
                    role=self.role,
                    path=SPARSE_PATH if sparse else DENSE_PATH,
                    backend=self.sparse_backend_name if sparse else self.dense_backend_name,
                    rank=_rank(),
                )
            )
        return sparse

    def forward_sparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        caps = self.capabilities
        name = self.sparse_backend_name
        if query.dtype not in caps.dtypes:
            raise ValueError(f"Sparse backend {name!r} does not support dtype {query.dtype}.")
        if attn_metadata is not None:
            if attn_metadata.attn_mask is not None and not caps.attention_mask:
                raise ValueError(f"Sparse backend {name!r} cannot apply attn_mask; the model emitted one.")
            if attn_metadata.full_attn_spans is not None and not caps.piecewise_spans:
                raise ValueError(f"Sparse backend {name!r} does not support piecewise attention spans.")
        return self.sparse_impl.forward(query, key, value, attn_metadata)
