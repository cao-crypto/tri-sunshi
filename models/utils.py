import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_ohem_loss(pred, target, keep_ratio=0.3):
    """
    实现Online Hard Example Mining (OHEM) Loss
    
    Args:
        pred: 预测值，Shape 可能是 `(B, C, N)` 或 `(B*N, C)`
        target: 真实标签，Shape 可能是 `(B, N)` 或 `(B*N)`
        keep_ratio: 保留难样本的比例，默认为 0.3
        
    Returns:
        ohem_loss: OHEM Loss值
    """
    # 处理3D输入 (B, C, N)
    if pred.dim() == 3:
        B, C, N = pred.shape
        pred = pred.transpose(1, 2).reshape(-1, C)  # 转换为 (B*N, C)
        target = target.reshape(-1)  # 转换为 (B*N)
    
    # 计算逐像素的CrossEntropy Loss
    # reduction='none' 确保返回每个点的Loss值
    loss = F.cross_entropy(pred, target, reduction='none')
    
    # 对Loss进行降序排序
    sorted_loss, _ = torch.sort(loss, descending=True)
    
    # 计算需要保留的样本数量
    keep_num = int(keep_ratio * len(sorted_loss))
    keep_num = max(1, keep_num)  # 确保至少保留一个样本
    
    # 只保留前keep_num个最大的Loss值
    ohem_loss = sorted_loss[:keep_num].mean()
    
    return ohem_loss


class PrototypeGuidedGating(nn.Module):
    """
    Task-Aware Prototype-Guided Cross-Gating Module
    Args:
        deep_dim: 深层特征维度 (C_d)
        shallow_dim: 浅层特征维度 (C_s)
    """

    def __init__(self, deep_dim=1024, shallow_dim=64):
        super(PrototypeGuidedGating, self).__init__()

        # 原型对齐：将深层原型映射到浅层维度
        self.proto_align = nn.Sequential(
            nn.Linear(deep_dim, shallow_dim),
            nn.BatchNorm1d(shallow_dim),
            nn.LeakyReLU(0.2)
        )

        # 门控生成：从相关性激活图生成门控
        self.gate_gen = nn.Sequential(
            nn.Conv1d(shallow_dim, shallow_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(shallow_dim),
            nn.Sigmoid()
        )

        # 残差融合：将筛选后的浅层特征与深层特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(deep_dim + shallow_dim, deep_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(deep_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, query_deep, query_shallow, support_proto):
        """
        Args:
            query_deep: 查询集深层特征，形状 [B, C_d, N]
            query_shallow: 查询集浅层特征，形状 [B, C_s, N]
            support_proto: 支持集原型，形状 [B, C_d]
        Returns:
            query_refined: 融合后的查询集特征，形状 [B, C_d, N]
        """
        B, C_d, N = query_deep.shape
        C_s = query_shallow.shape[1]

        # 原型对齐
        proto_key = self.proto_align(support_proto)  # [B, C_s]
        proto_key = proto_key.unsqueeze(2)  # [B, C_s, 1]

        # 相关性激活：Channel-wise Product
        activation = query_shallow * proto_key  # [B, C_s, N]

        # 门控生成
        gate = self.gate_gen(activation)  # [B, C_s, N]

        # 特征筛选
        refined_shallow = query_shallow * gate  # [B, C_s, N]

        # 残差融合
        fused = torch.cat([query_deep, refined_shallow], dim=1)  # [B, C_d + C_s, N]
        fused = self.fusion(fused)  # [B, C_d, N]
        query_refined = fused + query_deep  # 残差连接

        return query_refined