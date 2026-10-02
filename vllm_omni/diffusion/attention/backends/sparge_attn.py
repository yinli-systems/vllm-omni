# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Training-free block-sparse attention with SpargeAttn (thu-ml/SpargeAttn).

``spas_sage2_attn_meansim_topk_cuda`` selects, for every query block, the
``topk`` fraction of key blocks by mean-similarity, then runs the
SageAttention2 INT8/FP8 block-sparse kernel on the selected blocks.

SpargeAttn must launch its CUDA kernels on the current PyTorch stream
(thu-ml/SpargeAttn commit "Launch CUDA kernels on the current PyTorch stream").
Older builds launch them on the legacy default stream while their Triton
kernels use the current stream; vLLM workers run on a dedicated non-default
stream, so those builds race and fail with illegal memory accesses.

The implementation consumes the packed metadata dense backends let models
emit: a ``[real, pad]`` packing (``packed_padding`` or ``valid_kv_length``)
runs the kernel on the real rows and zero-fills the padding rows, and a
multi-document packing runs one kernel call per document. Anything else it
cannot execute exactly raises.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    SparseAdapterCapabilities,
)

logger = init_logger(__name__)

# The kernel tiles queries/keys in 64/128-row blocks and rejects shorter sequences.
MIN_SEQ_LEN = 128

_OPTION_DEFAULTS: dict[str, Any] = {
    "topk": 0.5,
    "simthreshd1": -0.1,
    "cdfthreshd": None,
    "pvthreshd": 50,
    "log_sparsity_every": 0,
}


def _sparge_kernel():
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

    return spas_sage2_attn_meansim_topk_cuda


def _parse_options(backend_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    options = dict(_OPTION_DEFAULTS)
    unknown = set(backend_kwargs or {}) - set(options)
    if unknown:
        raise ValueError(f"Unknown SPARGE_ATTN options: {sorted(unknown)}; supported: {sorted(options)}.")
    options.update(backend_kwargs or {})
    topk = options["topk"]
    if isinstance(topk, bool) or not isinstance(topk, int | float) or not 0.0 < topk <= 1.0:
        raise ValueError(f"SPARGE_ATTN topk must be in (0, 1]; got {topk!r}.")
    every = options["log_sparsity_every"]
    if type(every) is not int or every < 0:
        raise ValueError(f"SPARGE_ATTN log_sparsity_every must be an int >= 0; got {every!r}.")
    return options


class SpargeAttentionBackend(AttentionBackend):
    supported_platforms = ("cuda",)
    supports_prefix_kv_slicing = True

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        return True

    @classmethod
    def supports_multi_doc_packed_varlen(cls) -> bool:
        return True

    @classmethod
    def validate_available(cls) -> None:
        try:
            _sparge_kernel()
        except ImportError as e:
            raise ImportError(
                "SPARGE_ATTN requires SpargeAttn (https://github.com/thu-ml/SpargeAttn); "
                "build it against the installed torch with `pip install --no-build-isolation .`, from a "
                "revision that launches its CUDA kernels on the current PyTorch stream."
            ) from e

    @classmethod
    def get_sparse_adapter_capabilities(cls) -> SparseAdapterCapabilities:
        return SparseAdapterCapabilities(
            platforms=frozenset({"cuda"}),
            dtypes=frozenset({torch.bfloat16, torch.float16}),
            packed_mask_free=True,
            prefix_kv_slicing=True,
            multi_doc_packed_varlen=True,
            ulysses=True,
            option_keys=frozenset(_OPTION_DEFAULTS),
        )

    @staticmethod
    def get_name() -> str:
        return "SPARGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SpargeAttentionImpl]:
        return SpargeAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]


class SpargeAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        if causal:
            raise ValueError("SPARGE_ATTN does not support causal attention.")
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise ValueError("SPARGE_ATTN requires num_kv_heads == num_heads.")
        self.softmax_scale = softmax_scale
        self.prefix = prefix
        self.options = _parse_options(backend_kwargs)
        self._calls = 0
        self._sparsity_sum = 0.0
        self._sparsity_samples = 0

    @staticmethod
    def _documents(seq_len: int, attn_metadata: AttentionMetadata | None) -> list[tuple[int, int]]:
        """Return the [start, end) row ranges that hold real tokens."""
        if attn_metadata is None:
            return [(0, seq_len)]
        if attn_metadata.attn_mask is not None:
            raise ValueError("SPARGE_ATTN cannot apply attn_mask.")
        if attn_metadata.full_attn_spans is not None:
            raise ValueError("SPARGE_ATTN does not support piecewise attention spans.")
        padding = attn_metadata.packed_padding
        if padding is not None:
            if padding.q_length != padding.kv_length:
                raise ValueError("SPARGE_ATTN requires equal query and key valid lengths.")
            return [(0, padding.q_length)]
        cu_seqlens = attn_metadata.extra.get("cu_seqlens_q")
        valid = attn_metadata.extra.get("valid_kv_length")
        if cu_seqlens is not None and valid is not None and int(valid) < seq_len:
            # [real, pad] packing described by a valid prefix: one real document.
            return [(0, int(valid))]
        if cu_seqlens is not None:
            bounds = [int(b) for b in cu_seqlens.tolist()]
            if bounds[0] != 0 or bounds[-1] > seq_len or any(b > e for b, e in zip(bounds, bounds[1:])):
                raise ValueError(f"SPARGE_ATTN received invalid cu_seqlens {bounds} for length {seq_len}.")
            return [(start, end) for start, end in zip(bounds, bounds[1:]) if end > start]
        if valid is not None:
            return [(0, int(valid))]
        return [(0, seq_len)]

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        # Layout: [batch, seq, heads, head_dim] (NHD per batch item).
        documents = self._documents(query.shape[1], attn_metadata)
        if (len(documents) > 1 or documents[0] != (0, query.shape[1])) and query.shape[0] != 1:
            raise ValueError("SPARGE_ATTN packed layouts require batch size 1.")
        kernel = _sparge_kernel()
        opts = self.options
        self._calls += 1
        every = opts["log_sparsity_every"]
        want_sparsity = every > 0 and self._calls % every == 0
        covers_all = documents == [(0, query.shape[1])]
        out = torch.empty_like(query) if covers_all else torch.zeros_like(query)
        for start, end in documents:
            if end - start < MIN_SEQ_LEN:
                raise ValueError(f"SPARGE_ATTN needs at least {MIN_SEQ_LEN} rows per document; got {end - start}.")
            result = kernel(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                scale=self.softmax_scale,
                simthreshd1=opts["simthreshd1"],
                cdfthreshd=opts["cdfthreshd"],
                topk=opts["topk"],
                pvthreshd=opts["pvthreshd"],
                tensor_layout="NHD",
                output_dtype=query.dtype,
                return_sparsity=want_sparsity,
            )
            if want_sparsity:
                result, sparsity = result
                self._sparsity_sum += sparsity
                self._sparsity_samples += 1
            out[:, start:end] = result
        if want_sparsity:
            logger.info(
                "SPARGE_ATTN %s: mean block sparsity %.4f over %d sampled calls (topk=%s)",
                self.prefix,
                self._sparsity_sum / self._sparsity_samples,
                self._sparsity_samples,
                opts["topk"],
            )
        return out
