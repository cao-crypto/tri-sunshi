""" Prototypical Network
1. generate 3D & text prototypes: point_prototypes, text_prototypes
2. Aveage fusion 3D & text prototypes: fusion_prototypes
3. QGPA query-guided prorotype adaption: fusion_prototype_post
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import DGCNN
from models.dgcnn_new import DGCNN_semseg
from models.attention import *
from models.gmmn import GMMNnetwork,ProjectorNetwork
from einops import rearrange, repeat
# from torch_cluster import fps
from models.similarity_head import ShallowSimilarityHead, DeepSimilarityHead, LogitsFusion, PointWiseDynamicFusion
from models.utils import PrototypeGuidedGating
from models.backbone_adapters import get_backbone


class BoundaryAwareShallowBranch(nn.Module):
    """Boundary-aware shallow branch for enhanced local feature extraction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Edge-aware convolution
        self.edge_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Boundary confidence head
        self.boundary_head = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )
        
        # Feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv1d(64 + 1, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, shallow_feat, xyz):
        """
        Args:
            shallow_feat: [B, C, N] or [C, N]
            xyz: [B, N, 3] or [N, 3]
        Returns:
            refined_shallow_feat: [B, C_out, N] or [C_out, N]
            boundary_confidence: [B, 1, N] or [1, N]
        """
        squeeze_batch = False
        if shallow_feat.dim() == 2:
            shallow_feat = shallow_feat.unsqueeze(0)
            squeeze_batch = True
        if xyz.dim() == 2:
            xyz = xyz.unsqueeze(0)

        # xyz is currently kept for interface compatibility and future geometry-aware refinement.
        _ = xyz

        edge_feat = self.edge_conv(shallow_feat)
        boundary_conf = torch.sigmoid(self.boundary_head(edge_feat))
        combined = torch.cat([edge_feat, boundary_conf], dim=1)
        refined = self.refine_conv(combined)

        if squeeze_batch:
            refined = refined.squeeze(0)
            boundary_conf = boundary_conf.squeeze(0)

        return refined, boundary_conf


class MutualAggregationModule(nn.Module):
    """Mutual Aggregation Module (MAM) for bidirectional support-query interaction"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.ffn_s = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.ffn_q = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, support_feat, query_feat):
        """
        Args:
            support_feat: (n_way, k_shot, C, N)
            query_feat: (B, C, N)
        Returns:
            enhanced_support_feat: (n_way, k_shot, C, N)
            enhanced_query_feat: (B, C, N)
        """
        n_way, k_shot, C, N = support_feat.shape
        B = query_feat.shape[0]

        # Average support over k_shot to get per-way support
        support_feat_avg = support_feat.mean(dim=1)  # (n_way, C, N)

        # Reshape for multihead attention: (N, n_way, C) and (N, B, C)
        support_reshaped = support_feat_avg.permute(2, 0, 1)  # (N, n_way, C)
        query_reshaped = query_feat.permute(2, 0, 1)  # (N, B, C)

        # Compute cross-attention: query attends to support
        attn_output_q, _ = self.attention(query_reshaped, support_reshaped, support_reshaped)
        # Compute cross-attention: support attends to query
        attn_output_s, _ = self.attention(support_reshaped, query_reshaped, query_reshaped)

        # Residual connection and norm
        query_updated = self.norm_q(query_reshaped + attn_output_q)
        support_updated = self.norm_s(support_reshaped + attn_output_s)

        # FFN
        query_updated = self.ffn_q(query_updated)
        support_updated = self.ffn_s(support_updated)

        # Reshape back
        enhanced_query_feat = query_updated.permute(1, 2, 0)  # (B, C, N)
        enhanced_support_feat_avg = support_updated.permute(1, 2, 0)  # (n_way, C, N)

        # Expand back to k_shot
        enhanced_support_feat = enhanced_support_feat_avg.unsqueeze(1).repeat(1, k_shot, 1, 1)  # (n_way, k_shot, C, N)

        return enhanced_support_feat, enhanced_query_feat


class CommonalityBasedPrototypeSelection(nn.Module):
    """Commonality-based Prototype Selection (CPS) for semantic purification"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, support_feat, query_feat, fg_mask, bg_mask, n_way, k_shot):
        """
        Args:
            support_feat: (n_way, k_shot, C, N)
            query_feat: (B, C, N)
            fg_mask: (n_way, k_shot, N)
            bg_mask: (n_way, k_shot, N)
            n_way: number of ways
            k_shot: number of shots
        Returns:
            purified_prototypes: list of [bg_prototype] + fg_prototypes
        """
        # Generate initial prototypes
        fg_feat = support_feat * fg_mask.unsqueeze(2)
        bg_feat = support_feat * bg_mask.unsqueeze(2)

        fg_prototypes_init = []
        for way in range(n_way):
            sum_val = fg_mask[way].sum() + 1e-5
            # fg_feat[way] shape: (k_shot, C, N) - 3D tensor
            # Sum over k_shot (dim=0) and points (dim=2), keep feature dim (dim=1)
            fg_proto = fg_feat[way].sum(dim=(0, 2)) / sum_val
            fg_prototypes_init.append(fg_proto)

        bg_sum = bg_mask.sum() + 1e-5
        # bg_feat shape: (n_way, k_shot, C, N) - 4D tensor
        # Sum over n_way (dim=0), k_shot (dim=1), and points (dim=3), keep feature dim (dim=2)
        bg_prototype_init = bg_feat.sum(dim=(0, 1, 3)) / bg_sum

        # Compute commonality maps and purify prototypes
        fg_prototypes_purified = []
        for way in range(n_way):
            # Get query for this way (assuming B = n_way)
            q_feat = query_feat[way]  # (C, N)
            # Compute similarity between query and initial prototype
            # q_feat shape: (C, N), fg_prototypes_init[way].unsqueeze(1) shape: (C, 1)
            # Compute similarity along feature dim (dim=0)
            similarity = F.cosine_similarity(q_feat, fg_prototypes_init[way].unsqueeze(1), dim=0)  # (N,)
            # Normalize similarity to get weights
            weights = F.softmax(similarity, dim=0)  # (N,)
            # Apply weights to support features of this way
            # support_feat[way] shape: (k_shot, C, N), weights shape: (N,)
            # Unsqueeze weights to (1, 1, N) to broadcast with (k_shot, C, N)
            weighted_support = support_feat[way] * weights.unsqueeze(0).unsqueeze(1)  # (k_shot, C, N)
            # Multiply by fg_mask to only consider foreground points
            weighted_support = weighted_support * fg_mask[way].unsqueeze(1)  # (k_shot, C, N)
            # Compute purified prototype
            sum_val = fg_mask[way].sum() + 1e-5
            # weighted_support shape: (k_shot, C, N) - 3D tensor
            # Sum over k_shot (dim=0) and points (dim=2), keep feature dim (dim=1)
            fg_proto_purified = weighted_support.sum(dim=(0, 2)) / sum_val
            fg_prototypes_purified.append(fg_proto_purified)

        # Purify background prototype
        # Average similarity across all queries
        bg_similarities = []
        for b in range(query_feat.shape[0]):
            q_feat = query_feat[b]  # (C, N)
            # Compute similarity between query and initial prototype
            # q_feat shape: (C, N), bg_prototype_init.unsqueeze(1) shape: (C, 1)
            # Compute similarity along feature dim (dim=0)
            similarity = F.cosine_similarity(q_feat, bg_prototype_init.unsqueeze(1), dim=0)  # (N,)
            bg_similarities.append(similarity)
        bg_similarity = torch.stack(bg_similarities).mean(dim=0)  # (N,)
        bg_weights = F.softmax(bg_similarity, dim=0)  # (N,)

        # Apply weights to all support features
        # support_feat shape: (n_way, k_shot, C, N), bg_weights shape: (N,)
        # Unsqueeze bg_weights to (1, 1, 1, N) to broadcast with (n_way, k_shot, C, N)
        weighted_support_bg = support_feat * bg_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (n_way, k_shot, C, N)
        # Multiply by bg_mask to only consider background points
        weighted_support_bg = weighted_support_bg * bg_mask.unsqueeze(2)  # (n_way, k_shot, C, N)
        # Compute purified background prototype
        # weighted_support_bg shape: (n_way, k_shot, C, N) - 4D tensor
        # Sum over n_way (dim=0), k_shot (dim=1), and points (dim=3), keep feature dim (dim=2)
        bg_proto_purified = weighted_support_bg.sum(dim=(0, 1, 3)) / (bg_mask.sum() + 1e-5)

        # Combine prototypes
        purified_prototypes = [bg_proto_purified] + fg_prototypes_purified
        return purified_prototypes


class BaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i - 1]
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_dim, params[i], 1),
                nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs - 1:
                x = F.relu(x)
        return x



class GPCPR(nn.Module):
    def __init__(self, args):
        super(GPCPR, self).__init__()
        # self.args = args
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = 'cosine'
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.use_linear_proj = args.use_linear_proj
        self.use_supervise_prototype = args.use_supervise_prototype # SR loss
        self.use_align = args.use_align # align loss
        self.sr_weight = getattr(args, 'sr_weight', 1.0) # Semantic Regularization loss weight
        # Get backbone using adapter
        self.encoder = get_backbone(args)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        if self.use_linear_proj:
            self.conv_1 = nn.Sequential(nn.Conv1d(args.train_dim, args.train_dim, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(args.train_dim),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.use_transformer = args.use_transformer
        if self.use_transformer:
            self.transformer = QGPA()

        # GPCPR add
        self.use_text = args.use_text
        self.use_text_diff = args.use_text_diff
        if args.use_text or args.use_text_diff:
            self.text_projector = ProjectorNetwork(args.noise_dim, args.train_dim, args.train_dim, args.gmm_dropout)
        if args.use_text:
            self.text_compressor = nn.MultiheadAttention(embed_dim=args.train_dim, num_heads=4, dropout=0.5)
        if args.use_text_diff:
            self.text_compressor_diff = nn.MultiheadAttention(embed_dim=args.train_dim, num_heads=4, dropout=0.5)

        self.use_pcpr=args.use_pcpr
        if args.use_pcpr:
            self.proto_compressor = MultiHeadAttention(in_channel=args.train_dim, out_channel=args.train_dim, n_heads=4,att_dropout=0.5, use_proj=False)

        self.use_dd_loss = args.use_dd_loss   #dd-loss
        self.dd_ratio1 = args.dd_ratio1
        self.dd_ratio2 = args.dd_ratio2

        sim_dim = args.train_dim
        if sim_dim is None:
            # 常见命名兜底：emb_dims / feat_dim 等（按你仓库实际字段调整）
            sim_dim = getattr(args, "emb_dims", None) or getattr(args, "feat_dim", None)
        if sim_dim is None:
            raise ValueError("Cannot infer sim_dim. Please set args.sim_dim to your feature dim C.")

        # Fusion configuration
        self.fusion_mode = getattr(args, "fusion_mode", "scalar")
        self.sim_head = nn.ModuleDict({
            "ssm": ShallowSimilarityHead(
                dim=sim_dim,
                num_heads=getattr(args, "ssm_heads", 4),
                init_scale=getattr(args, "ssm_init_scale", 10.0),
                attn_dropout=getattr(args, "ssm_attn_dropout", 0.0),
                proj_dropout=getattr(args, "ssm_proj_dropout", 0.0),
            ),
            "dsm": DeepSimilarityHead(
                dim=sim_dim,
                depth=getattr(args, "dsm_depth", 2),
                num_heads=getattr(args, "dsm_heads", 4),
                ffn_ratio=getattr(args, "dsm_ffn_ratio", 4.0),
                dropout=getattr(args, "dsm_dropout", 0.0),
                init_scale=getattr(args, "dsm_init_scale", 10.0),
            ),
            "fusion": LogitsFusion(init_alpha=getattr(args, "fusion_alpha", 0.5)),
        })

        # Add dynamic fusion if needed
        if self.fusion_mode == "dynamic":
            self.sim_head["dynamic_fusion"] = PointWiseDynamicFusion(
                dim=sim_dim,
                num_classes=self.n_way + 1
            )

        # Boundary-aware shallow branch
        self.use_boundary_shallow = getattr(args, "use_boundary_shallow", False)
        if self.use_boundary_shallow:
            self.boundary_branch = BoundaryAwareShallowBranch(
                in_channels=64,
                out_channels=sim_dim
            )
            self.lambda_boundary = getattr(args, "lambda_boundary", 0.1)
            self.boundary_knn_k = getattr(args, "boundary_knn_k", 5)

        # 添加任务感知的原型引导交叉门控模块
        # 计算正确的deep_dim值，使其与task_proto_expanded的实际维度匹配
        # 根据getFeatures方法的实现，task_proto_expanded的维度是320
        self.fusion = PrototypeGuidedGating(deep_dim=320, shallow_dim=64)

        # MAM and CPS modules
        self.use_mam = args.use_mam
        self.use_cps = args.use_cps
        if self.use_mam:
            self.mam = MutualAggregationModule(dim=args.train_dim, num_heads=4, dropout=0.1)
        if self.use_cps:
            self.cps = CommonalityBasedPrototypeSelection(dim=args.train_dim)

    def forward(self, support_x, support_y, query_x, query_y, text_emb=None, text_emb_diff=None):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 1, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        # get features
        support_x = support_x.view(self.n_way * self.k_shot, self.in_channels, self.n_points)
        if self.use_attention:
            support_feat, support_xyz, support_shallow = self.getFeatures(support_x)
        else:
            support_feat, support_shallow = self.getFeatures(support_x)
            support_xyz = support_x[:, :3, :].transpose(1, 2)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)

        if self.use_attention:
            query_feat, query_xyz, query_shallow = self.getFeatures(query_x)
        else:
            query_feat, query_shallow = self.getFeatures(query_x)
            query_xyz = query_x[:, :3, :].transpose(1, 2)

        # Boundary-aware shallow branch
        boundary_loss = 0
        if self.use_boundary_shallow:
            support_shallow, _ = self.boundary_branch(support_shallow, support_xyz)
            query_shallow, query_boundary = self.boundary_branch(query_shallow, query_xyz)

            if self.training:
                query_boundary_labels = self.generate_boundary_labels(query_y, query_xyz)
                boundary_loss = F.binary_cross_entropy(query_boundary.squeeze(1), query_boundary_labels)

        # Reshape support shallow features
        support_shallow = support_shallow.view(self.n_way, self.k_shot, -1, self.n_points)

        # Mutual Aggregation Module (MAM)
        if self.use_mam:
            support_feat, query_feat = self.mam(support_feat, query_feat)

        # get bg/fg features: Fs'=Fs*Ms
        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        # Commonality-based Prototype Selection (CPS)
        if self.use_cps:
            # Apply CPS to purify prototypes
            purified_prototypes = self.cps(support_feat, query_feat, fg_mask, bg_mask, self.n_way, self.k_shot)
            prototypes = torch.stack(purified_prototypes, dim=0)
        else:
            # Original prototype generation
            fg_prototypes, bg_prototype = self.getPrototype(self.getMaskedFeatures(support_feat, fg_mask), self.getMaskedFeatures(support_feat, bg_mask))
            prototypes = [bg_prototype] + fg_prototypes
            prototypes = torch.stack(prototypes, dim=0)
        # 任务感知的原型引导交叉门控
        # 使用平均原型作为任务原型
        task_proto = prototypes.mean(dim=0)  # [C_d]
        # 扩展维度以匹配查询批次
        B = query_feat.shape[0]
        task_proto_expanded = task_proto.unsqueeze(0).repeat(B, 1)  # [B, C_d]
        # 应用融合模块
        query_refined = self.fusion(query_feat, query_shallow, task_proto_expanded)

        # save multi-stage results
        tep_proto = {}
        tep_pred = {}
        if self.use_dd_loss:
            tep_proto['orig'] = (prototypes.unsqueeze(0).repeat(query_refined.shape[0], 1, 1))
            tep_pred['orig'] = (torch.stack(
                [self.calculateSimilarity(query_refined, prototype, self.dist_method) for prototype in prototypes],
                dim=1))

        # GCPR - diverse text
        if self.use_text and text_emb is not None:
            text_emb = self.text_projector(text_emb)   # [3, num, dim]
            prototypes = prototypes.unsqueeze(1)+self.text_compressor(prototypes.unsqueeze(1), text_emb, text_emb,need_weights=False)[0] # (out,attn)
            prototypes = prototypes.squeeze(1)  # [3,320]
            # prototypes = prototypes.unsqueeze(1)+self.text_compressor(prototypes.unsqueeze(1).transpose(0,1), text_emb.transpose(0,1), text_emb.transpose(0,1),need_weights=False)[0].transpose(0,1) # (out,attn)
            # prototypes = prototypes.squeeze(1)  # [3,320]
            if self.use_dd_loss:
                tep_proto['text']=(prototypes.unsqueeze(0).repeat(query_feat.shape[0], 1, 1))
                tep_pred['text']=(torch.stack([self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes],dim=1))
        # GCPR - differentiated text
        if self.use_text_diff and text_emb_diff is not None:
            text_emb_diff = self.text_projector(text_emb_diff)   # [3, num, dim]
            prototypes = prototypes.unsqueeze(1)+self.text_compressor_diff(prototypes.unsqueeze(1), text_emb_diff, text_emb_diff,need_weights=False)[0] # (out,attn)
            prototypes = prototypes.squeeze(1)  # [3,320]
            # prototypes = prototypes.unsqueeze(1)+self.text_compressor_diff(prototypes.unsqueeze(1).transpose(0,1), text_emb_diff.transpose(0,1), text_emb_diff.transpose(0,1),need_weights=False)[0].transpose(0,1) # (out,attn)
            # prototypes = prototypes.squeeze(1)  # [3,320]
            if self.use_dd_loss:
                tep_proto['text_diff']=(prototypes.unsqueeze(0).repeat(query_feat.shape[0], 1, 1))
                tep_pred['text_diff']=(torch.stack([self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes],dim=1))

        # Semantic Regularization (SR) loss: Support Self-Alignment
        sr_loss = 0
        if self.use_supervise_prototype:
            # Use the same prototypes for SR loss as used for query prediction
            if self.use_transformer and 'prototypes_all_post' in locals():
                # Use adapted prototypes from QGPA for SR loss
                sr_loss = self.semantic_regularization_loss(prototypes_all_post, support_feat, fg_mask, bg_mask, True)
            else:
                # Use purified prototypes from CPS for SR loss
                sr_loss = self.semantic_regularization_loss(prototypes, support_feat, fg_mask, bg_mask, False)


        if self.use_transformer:   # QGPA & loss Lseg
            prototypes_all = prototypes.unsqueeze(0).repeat(query_refined.shape[0], 1, 1)  # [2,3,320]
            support_feat_ = support_feat.mean(1)  # [2, 320, 2048]
            prototypes_all_post = self.transformer(query_refined, support_feat_, prototypes_all)

            # 注释掉了
            # prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
            # similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for
            #               prototype in prototypes_new]
            # query_pred = torch.stack(similarity, dim=1)
            # ================== SSM + DSM similarity head (transformer branch) ==================

            # 添加
            #q_feat = query_feat.transpose(1, 2).contiguous()
            # query_refined: [B, C, Nq] -> [B, Nq, C]
            q_feat = query_refined.transpose(1, 2).contiguous()

            # prototypes_all_post: [B, K, C]
            proto = prototypes_all_post

            # Shallow Similarity Module (SSM)
            logits_s = self.sim_head["ssm"](q_feat, proto)  # [B, Nq, K]

            # Deep Similarity Module (DSM)
            logits_d = self.sim_head["dsm"](q_feat, proto)  # [B, Nq, K]

            # Fusion
            if self.fusion_mode == "dynamic":
                # Get shallow and deep features for dynamic fusion
                # For shallow features, use the refined shallow features
                shallow_feat_for_fusion = query_shallow.transpose(1, 2).contiguous()  # [B, Nq, C]
                # For deep features, use the query features
                deep_feat_for_fusion = q_feat  # [B, Nq, C]
                
                logits_final, alpha = self.sim_head["dynamic_fusion"](
                    logits_s, logits_d, shallow_feat_for_fusion, deep_feat_for_fusion
                )
            else:
                logits_final = self.sim_head["fusion"](logits_s, logits_d)  # [B, Nq, K]

            # keep original interface: [B, K, Nq]
            query_pred = logits_final.permute(0, 2, 1).contiguous()

            if self.use_dd_loss:
                tep_proto['qgpa']=(prototypes_all_post)
                tep_pred['qgpa']=(query_pred)

            if self.use_pcpr:
                query_bg_fg_features = self.extract_query_features(query_feat, query_pred)  # (n_way+1, kp, d)
                spt_prototypes = prototypes_all_post.transpose(0, 1)
                qry_bg_prototypes = self.proto_compressor([spt_prototypes[:1], query_bg_fg_features[:1], query_bg_fg_features[:1]])  # (n_way, n_proto, d)
                qry_fg_prototypes = self.proto_compressor([spt_prototypes[1:], query_bg_fg_features[1:],query_bg_fg_features[1:]])  # (n_way, n_proto, d)
                prototypes_all_post = torch.cat([qry_bg_prototypes, qry_fg_prototypes], dim=0).transpose(0, 1)
                prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
                similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for
                              prototype in prototypes_new]
                query_pred = torch.stack(similarity, dim=1)
                if self.use_dd_loss:
                    tep_proto['pqmqm']=(prototypes_all_post)
                    tep_pred['pqmqm']=(query_pred)
            loss = self.computeCrossEntropyLoss(query_pred, query_y)
        else:
            # ================== SSM + DSM similarity head (no transformer branch) ==================

            # query_feat: [B, C, Nq] -> [B, Nq, C]
            q_feat = query_feat.transpose(1, 2).contiguous()

            # prototypes: [K, C] -> [B, K, C]
            proto = prototypes.unsqueeze(0).expand(q_feat.shape[0], -1, -1).contiguous()

            # SSM
            logits_s = self.sim_head["ssm"](q_feat, proto)  # [B, Nq, K]

            # DSM
            logits_d = self.sim_head["dsm"](q_feat, proto)  # [B, Nq, K]

            # Fusion
            if self.fusion_mode == "dynamic":
                # Get shallow and deep features for dynamic fusion
                shallow_feat_for_fusion = query_shallow.transpose(1, 2).contiguous()  # [B, Nq, C]
                deep_feat_for_fusion = q_feat  # [B, Nq, C]
                
                logits_final, alpha = self.sim_head["dynamic_fusion"](
                    logits_s, logits_d, shallow_feat_for_fusion, deep_feat_for_fusion
                )
            else:
                logits_final = self.sim_head["fusion"](logits_s, logits_d)

            # [B, K, Nq] for loss & evaluation
            query_pred = logits_final.permute(0, 2, 1).contiguous()

            loss = self.computeCrossEntropyLoss(query_pred, query_y)   # segmentation loss

        align_loss = 0
        if self.use_align:
            align_loss = align_loss + self.alignLoss_trans(query_feat, query_pred, support_feat, fg_mask, bg_mask)

        # Auxiliary loss for MAM to prevent feature drift
        mam_loss = 0
        if self.use_mam:
            # Compute feature consistency loss between original and enhanced features
            # This helps prevent the MAM from drifting too far from the original features
            original_support_feat = support_feat.view(self.n_way * self.k_shot, -1, self.n_points)
            original_query_feat = query_feat
            
            # Get original features (before MAM)
            if self.use_attention:
                original_support_feat, _, _ = self.getFeatures(support_x)
                original_query_feat, _, _ = self.getFeatures(query_x)
            else:
                original_support_feat, _ = self.getFeatures(support_x)
                original_query_feat, _ = self.getFeatures(query_x)
            
            original_support_feat = original_support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
            
            # Compute L2 loss between original and enhanced features
            mam_loss = F.mse_loss(support_feat, original_support_feat) + F.mse_loss(query_feat, original_query_feat)

        dd_loss = 0
        if self.use_dd_loss and self.use_pcpr and self.use_transformer:
            kl = torch.nn.KLDivLoss()
            T = 2
            keys = list(tep_proto.keys())
            if 'qgpa' in keys and 'pqmqm' in keys:
                dd_loss = dd_loss + self.dd_ratio1 * kl(F.log_softmax(tep_proto['qgpa'] / T, dim=-1),
                                                             F.softmax(tep_proto['pqmqm'].detach() / T, dim=-1)) * T * T  # [2, 3, 320]
                dd_loss = dd_loss + self.dd_ratio2 * kl(F.log_softmax(tep_pred['qgpa'] / T, dim=-2),
                                                          F.softmax(tep_pred['pqmqm'].detach() / T, dim=-2)) * T * T  # [2, 3, 2048]
            if 'text' in keys and 'text_diff' in keys:
                dd_loss = dd_loss + self.dd_ratio1 * kl(F.log_softmax(tep_proto['text'] / T, dim=-1),
                                                         F.softmax(tep_proto['text_diff'].detach() / T, dim=-1)) * T * T  # [2, 3, 320]

        # Add boundary loss if enabled
        total_loss = loss + align_loss + sr_loss * self.sr_weight + dd_loss + mam_loss
        if self.use_boundary_shallow:
            total_loss += boundary_loss * self.lambda_boundary

        return query_pred, total_loss


    def semantic_regularization_loss(self, prototypes, support_feat, fg_mask, bg_mask, use_transformer):
        """
        Semantic Regularization (SR) loss: Support Self-Alignment
        Ensures that prototypes can correctly segment the support set itself

        Args:
            prototypes: prototypes used for prediction (either from CPS or QGPA)
                shape: (K, C) if not using transformer, (B, K, C) if using transformer
            support_feat: support features
                shape: (n_way, k_shot, C, N)
            fg_mask: foreground masks for support images
                shape: (n_way, k_shot, N)
            bg_mask: background masks for support images
                shape: (n_way, k_shot, N)
            use_transformer: whether prototypes are from transformer (QGPA)
        """
        n_ways, n_shots = self.n_way, self.k_shot
        loss = 0

        for way in range(n_ways):
            for shot in range(n_shots):
                # Get support features for this way and shot
                img_fts = support_feat[way, shot].unsqueeze(0)  # (1, C, N)
                
                # Get prototypes: background + current way foreground
                if use_transformer:
                    # For transformer, prototypes have shape (B, K, C), take first batch
                    bg_proto = prototypes[0, 0]
                    fg_proto = prototypes[0, way + 1]
                else:
                    # For non-transformer, prototypes have shape (K, C)
                    bg_proto = prototypes[0]
                    fg_proto = prototypes[way + 1]
                
                # Calculate similarity using the same method as query set
                # Use the same similarity head approach
                img_fts_transposed = img_fts.transpose(1, 2).contiguous()  # (1, N, C)
                proto = torch.stack([bg_proto, fg_proto], dim=0).unsqueeze(0)  # (1, 2, C)
                
                # Shallow Similarity Module (SSM)
                logits_s = self.sim_head["ssm"](img_fts_transposed, proto)  # (1, N, 2)
                
                # Deep Similarity Module (DSM)
                logits_d = self.sim_head["dsm"](img_fts_transposed, proto)  # (1, N, 2)
                
                # Fusion
                logits_final = self.sim_head["fusion"](logits_s, logits_d)  # (1, N, 2)
                
                # Reshape for loss: (1, 2, N)
                supp_pred = logits_final.permute(0, 2, 1).contiguous()
                
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fg_mask[way, shot], 255, device=img_fts.device).long()
                supp_label[fg_mask[way, shot] == 1] = 1  # foreground
                supp_label[bg_mask[way, shot] == 1] = 0  # background
                
                # Compute Cross-Entropy loss
                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        
        return loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features.

        Args:
            x: input data with shape (B, C_in, L)
        Returns:
            if use_attention:
                feat: (B, C_out, L), xyz: (B, L, 3), shallow_feat: (B, C_s, L)
            else:
                feat: (B, C_out, L), shallow_feat: (B, C_s, L)
        """
        enc_out = self.encoder(x)
        if not isinstance(enc_out, dict):
            raise TypeError(f"encoder output must be a dict, got {type(enc_out)}")

        feat_level2 = enc_out['final_feat']
        x_shallow = enc_out['shallow_feat']
        xyz = enc_out['xyz']
        multi_scale_feats = enc_out.get('multi_scale_feats', [])
        if isinstance(multi_scale_feats, torch.Tensor):
            multi_scale_feats = [multi_scale_feats]
        elif multi_scale_feats is None:
            multi_scale_feats = []
        else:
            multi_scale_feats = list(multi_scale_feats)

        if not isinstance(feat_level2, torch.Tensor):
            raise TypeError(f"final_feat must be a tensor, got {type(feat_level2)}")
        if not isinstance(x_shallow, torch.Tensor):
            raise TypeError(f"shallow_feat must be a tensor, got {type(x_shallow)}")
        if not isinstance(xyz, torch.Tensor):
            raise TypeError(f"xyz must be a tensor, got {type(xyz)}")

        if xyz.dim() != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must have shape [B, N, 3], got {tuple(xyz.shape)}")
        if feat_level2.dim() != 3:
            raise ValueError(f"final_feat must have shape [B, C, N], got {tuple(feat_level2.shape)}")
        if x_shallow.dim() != 3:
            raise ValueError(f"shallow_feat must have shape [B, C, N], got {tuple(x_shallow.shape)}")

        if len(multi_scale_feats) == 0:
            multi_scale_feats = [x_shallow, feat_level2]

        feat_level3 = self.base_learner(feat_level2)

        if self.use_attention:
            local_feats = list(multi_scale_feats[:3])
            while len(local_feats) < 3:
                local_feats.append(local_feats[-1] if len(local_feats) > 0 else feat_level2)

            att_feat = self.att_learner(feat_level2)
            fused = torch.cat((local_feats[0], local_feats[1], local_feats[2], att_feat, feat_level3), dim=1)
            if self.use_linear_proj:
                fused = self.conv_1(fused)
            return fused, xyz, x_shallow

        local_feat = multi_scale_feats[0]
        map_feat = self.linear_mapper(feat_level2)
        return torch.cat((local_feat, map_feat, feat_level3), dim=1), x_shallow

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0, 1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype


    def calculateSimilarity(self, feat, prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':  # prototype[None, ..., None] [1, 320, 1]
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2) ** 2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        return similarity



    def calculateSimilarity_trans(self, feat, prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[..., None], p=2) ** 2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        return similarity

    def calculateSimilarity_trans(self, feat, prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[..., None], p=2) ** 2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels, keep_ratio=0.3):
        """ Calculate the OHEM Loss for query set
        """
        from models.utils import calc_ohem_loss
        return calc_ohem_loss(query_logits, query_labels, keep_ratio=keep_ratio)

    def extract_query_features(self, qry_fts, pred):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x num_points
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_feature = (qry_fts.unsqueeze(1) * pred_mask)

        return rearrange(qry_fts.unsqueeze(1) * pred_mask,'k n d p -> n (k p) d')

    def generate_boundary_labels(self, labels, xyz):
        """
        Generate boundary labels based on kNN neighbors.
        A point is a boundary point if it has at least one neighbor with a different label.
        
        Args:
            labels: [B, N]
            xyz: [B, N, 3]
        Returns:
            boundary_labels: [B, N]
        """
        B, N = labels.shape
        boundary_labels = torch.zeros_like(labels, dtype=torch.float32, device=labels.device)
        
        for b in range(B):
            # Compute pairwise distances
            xyz_b = xyz[b]
            dist = torch.cdist(xyz_b, xyz_b)
            
            # Get k nearest neighbors
            _, idx = dist.topk(self.boundary_knn_k + 1, dim=1, largest=False)
            idx = idx[:, 1:]  # Exclude self
            
            # Get neighbor labels
            neighbor_labels = labels[b][idx]
            
            # Check if any neighbor has different label
            boundary = torch.any(neighbor_labels != labels[b].unsqueeze(1), dim=1)
            boundary_labels[b] = boundary.float()
        
        return boundary_labels

    def alignLoss_trans(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x num_points
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'

        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
        # print('qry_prototypes shape',qry_prototypes.shape)   # [3,320]
        # print('text_prototypes shape',text_prototypes.shape)   #[2,3,320]
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)
                prototypes_all = torch.stack(prototypes, dim=0).unsqueeze(0)
                prototypes_all_post = self.transformer(img_fts, qry_fts.mean(0).unsqueeze(0), prototypes_all)
                prototypes_new = [prototypes_all_post[0, 0], prototypes_all_post[0, 1]]

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in
                             prototypes_new]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss