"""SSM + DSM similarity head for few-shot 3D point cloud segmentation.

Recommended tensor shapes:
- query_feat: [B, Nq, C]
- proto:      [B, K,  C]   (K is number of classes; often n_way+1 including background)
- logits:     [B, Nq, K]

Notes:
- If you have query_feat as [B, C, Nq], pass `query_feat.transpose(1, 2)` before calling.
- If you have proto as [K, C], pass `proto.unsqueeze(0).expand(B, -1, -1)` before calling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _assert_3d(x: torch.Tensor, name: str) -> None:
    if x.dim() != 3:
        raise ValueError(f"{name} must be a 3D tensor, got shape={tuple(x.shape)}")


def _ensure_query_proto_shapes(
    query_feat: torch.Tensor,
    proto: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure shapes are [B, Nq, C] and [B, K, C].

    Accepts:
    - query_feat as [B, C, Nq] (common in point cloud backbones)
    - proto as [1, K, C] for broadcasting across B

    Returns:
        query_feat_bnc, proto_bkc
    """
    _assert_3d(query_feat, "query_feat")
    _assert_3d(proto, "proto")

    # If query_feat is [B, C, Nq] and proto is [B, K, C], then proto.shape[-1] == C.
    # In that case, query_feat.shape[-1] != C and we need transpose.
    if query_feat.shape[-1] != proto.shape[-1]:
        # Assume [B, C, Nq] -> [B, Nq, C]
        query_feat = query_feat.transpose(1, 2).contiguous()

    # Broadcast proto if it is [1, K, C]
    if proto.shape[0] == 1 and query_feat.shape[0] > 1:
        proto = proto.expand(query_feat.shape[0], -1, -1).contiguous()

    if proto.shape[0] != query_feat.shape[0]:
        raise ValueError(
            f"Batch size mismatch: query_feat B={query_feat.shape[0]} vs proto B={proto.shape[0]}"
        )

    if query_feat.shape[-1] != proto.shape[-1]:
        raise ValueError(
            "Feature dim mismatch after normalization: "
            f"query_feat {tuple(query_feat.shape)} vs proto {tuple(proto.shape)}"
        )

    return query_feat, proto


class ShallowSimilarityHead(nn.Module):
    """SSM: Cross-attention (Q=query, K/V=proto) + normalized dot-product logits."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        init_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
        )
        self.proj_drop = nn.Dropout(proj_dropout)

        # Multiplicative temperature (scale). Unconstrained is common and works well.
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, query_feat: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        """Return logits_s with shape [B, Nq, K]."""
        query_feat, proto = _ensure_query_proto_shapes(query_feat, proto)

        # Cross-attention: each query token attends over class prototypes.
        # fq_att, _ = self.mha(query=query_feat, key=proto, value=proto, need_weights=False)

        # query_feat: [B, Nq, C]
        # proto:      [B, K, C]

        # ---- transpose to MultiheadAttention expected format ----
        # [B, N, C] -> [N, B, C]
        q = query_feat.transpose(0, 1).contiguous()  # [Nq, B, C]
        k = proto.transpose(0, 1).contiguous()  # [K,  B, C]
        v = k  # [K,  B, C]

        # ---- cross-attention ----
        fq_att, _ = self.mha(
            q, k, v,
            need_weights=False
        )  # [Nq, B, C]

        # ---- transpose back ----
        fq_att = fq_att.transpose(0, 1).contiguous()  # [B, Nq, C]

        fq_att = self.proj_drop(fq_att)

        q = F.normalize(fq_att, p=2, dim=-1)
        p = F.normalize(proto, p=2, dim=-1)

        # [B, Nq, K]
        logits = torch.einsum("bnc,bkc->bnk", q, p) * self.logit_scale
        return logits


class DeepSimilarityHead(nn.Module):
    """DSM: Joint Transformer encoder over [proto; query] tokens + normalized dot-product logits."""

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 4,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        init_scale: float = 10.0,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        ffn_dim = int(dim * ffn_ratio)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, query_feat: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        """Return logits_d with shape [B, Nq, K]."""

        B, Nq, C = query_feat.shape
        K = proto.shape[1]

        # ---- concat tokens ----
        x = torch.cat([proto, query_feat], dim=1)     # [B, K+Nq, C]

        # ---- transpose to (seq, batch, dim) ----
        x = x.transpose(0, 1).contiguous()             # [K+Nq, B, C]

        # ---- transformer encoder ----
        x = self.encoder(x)                            # [K+Nq, B, C]

        # ---- transpose back ----
        x = x.transpose(0, 1).contiguous()             # [B, K+Nq, C]

        proto_ref = x[:, :K, :]                         # [B, K, C]
        query_ref = x[:, K:, :]                         # [B, Nq, C]

        # ---- cosine similarity ----
        proto_ref = F.normalize(proto_ref, dim=-1)
        query_ref = F.normalize(query_ref, dim=-1)

        logits = torch.einsum("bnc,bkc->bnk", query_ref, proto_ref)
        logits = logits * self.logit_scale

        return logits

        # query_feat, proto = _ensure_query_proto_shapes(query_feat, proto)
        # B, Nq, _ = query_feat.shape
        # K = proto.shape[1]
        #
        # # [B, K+Nq, C]
        # x = torch.cat([proto, query_feat], dim=1)
        # x_ref = self.encoder(x)
        #
        # refined_protos = x_ref[:, :K, :]
        # refined_queries = x_ref[:, K:, :]
        #
        # q = F.normalize(refined_queries, p=2, dim=-1)
        # p = F.normalize(refined_protos, p=2, dim=-1)
        #
        # logits = torch.einsum("bnc,bkc->bnk", q, p) * self.logit_scale
        # return logits


class PointWiseDynamicFusion(nn.Module):
    """Point-wise dynamic fusion: logits = alpha * logits_s + (1 - alpha) * logits_d,
    where alpha is computed per point based on shallow and deep features.
    """

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        # MLP to compute point-wise attention weights
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + num_classes * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, logits_s: torch.Tensor, logits_d: torch.Tensor, feat_s: torch.Tensor, feat_d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_s: shallow logits [B, Nq, K]
            logits_d: deep logits [B, Nq, K]
            feat_s: shallow features [B, Nq, C]
            feat_d: deep features [B, Nq, C]
        Returns:
            fused_logits: [B, Nq, K]
        """
        if logits_s.shape != logits_d.shape:
            raise ValueError(
                f"logits_s shape {tuple(logits_s.shape)} != logits_d shape {tuple(logits_d.shape)}"
            )

        B, Nq, K = logits_s.shape
        C = feat_s.shape[-1]

        # Compute point-wise features for fusion
        feat_diff = feat_s - feat_d
        combined = torch.cat([feat_s, feat_d, feat_diff, logits_s, logits_d], dim=-1)
        alpha = torch.sigmoid(self.mlp(combined))  # [B, Nq, 1]
        fused_logits = alpha * logits_s + (1.0 - alpha) * logits_d
        return fused_logits, alpha


class LogitsFusion(nn.Module):
    """Learnable fusion: logits = alpha * logits_s + (1 - alpha) * logits_d."""

    def __init__(self, init_alpha: float = 0.5) -> None:
        super().__init__()
        init_alpha = float(init_alpha)
        init_alpha = min(max(init_alpha, 1e-4), 1.0 - 1e-4)
        init_logit = torch.log(torch.tensor(init_alpha / (1.0 - init_alpha)))
        self.alpha_logit = nn.Parameter(init_logit)

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, logits_s: torch.Tensor, logits_d: torch.Tensor) -> torch.Tensor:
        if logits_s.shape != logits_d.shape:
            raise ValueError(
                f"logits_s shape {tuple(logits_s.shape)} != logits_d shape {tuple(logits_d.shape)}"
            )
        a = self.alpha()
        return a * logits_s + (1.0 - a) * logits_d