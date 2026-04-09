"""Backbone adapters for different backbones to return unified output structure."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import DGCNN
from models.dgcnn_new import DGCNN_semseg


class BackboneAdapter(nn.Module):
    """Base class for backbone adapters."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        """Return unified output structure."""
        raise NotImplementedError


class DGCNNAdapter(BackboneAdapter):
    """Adapter for DGCNN-style backbones.

    Always returns a dict with keys:
        - final_feat: [B, C, N]
        - shallow_feat: [B, C_s, N]
        - xyz: [B, N, 3]
        - multi_scale_feats: list[tensor]
    """

    @staticmethod
    def _default_xyz(x):
        return x[:, :3, :].transpose(1, 2).contiguous()

    @staticmethod
    def _as_feat_list(feats):
        if feats is None:
            return []
        if isinstance(feats, torch.Tensor):
            return [feats]
        if isinstance(feats, (list, tuple)):
            return [f for f in feats if isinstance(f, torch.Tensor)]
        raise ValueError(f"Unsupported multi-scale feature type: {type(feats)}")

    @staticmethod
    def _normalize_dict_output(output, x):
        final_feat = output.get('final_feat', output.get('out', output.get('feat_level2', output.get('feat'))))
        multi_scale_feats = output.get('multi_scale_feats', output.get('edgeconv_outputs', output.get('feat_level1')))
        multi_scale_feats = DGCNNAdapter._as_feat_list(multi_scale_feats)

        shallow_feat = output.get('shallow_feat', output.get('x_shallow'))
        if shallow_feat is None:
            shallow_feat = multi_scale_feats[0] if len(multi_scale_feats) > 0 else final_feat

        xyz = output.get('xyz', DGCNNAdapter._default_xyz(x))

        if final_feat is None:
            raise KeyError('Backbone output dict is missing final_feat/out/feat_level2/feat')
        if shallow_feat is None:
            raise KeyError('Backbone output dict is missing shallow_feat/x_shallow and no fallback is available')

        return {
            'final_feat': final_feat,
            'shallow_feat': shallow_feat,
            'xyz': xyz,
            'multi_scale_feats': multi_scale_feats,
        }

    @staticmethod
    def _normalize_legacy_output(output, x):
        xyz = DGCNNAdapter._default_xyz(x)

        if len(output) == 4:
            edgeconv_outputs, final_feat, maybe_xyz, shallow_feat = output
            if isinstance(maybe_xyz, torch.Tensor) and maybe_xyz.dim() == 3 and maybe_xyz.shape[-1] == 3:
                xyz = maybe_xyz
        elif len(output) == 3:
            first, second, third = output
            edgeconv_outputs = first
            final_feat = second
            if isinstance(third, torch.Tensor) and third.dim() == 3 and third.shape[-1] == 3:
                shallow_feat = None
                xyz = third
            else:
                shallow_feat = third
        elif len(output) == 2:
            edgeconv_outputs, final_feat = output
            shallow_feat = None
        else:
            raise ValueError(f"Unexpected legacy output format from DGCNN: {len(output)} elements")

        multi_scale_feats = DGCNNAdapter._as_feat_list(edgeconv_outputs)
        if shallow_feat is None:
            shallow_feat = multi_scale_feats[0] if len(multi_scale_feats) > 0 else final_feat

        return {
            'final_feat': final_feat,
            'shallow_feat': shallow_feat,
            'xyz': xyz,
            'multi_scale_feats': multi_scale_feats,
        }

    def forward(self, x):
        output = self.backbone(x)

        if isinstance(output, dict):
            return self._normalize_dict_output(output, x)

        if isinstance(output, (tuple, list)):
            return self._normalize_legacy_output(output, x)

        raise ValueError(f"Unexpected output type from DGCNN: {type(output)}")


class PTv3Adapter(BackboneAdapter):
    """Adapter for PTv3 backbone."""

    def __init__(self):
        # Placeholder for PTv3 backbone
        super().__init__(nn.Identity())
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

    def forward(self, x):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        x_shallow = F.relu(self.conv1(x))
        x_mid = F.relu(self.conv2(x_shallow))
        x_deep = F.relu(self.conv3(x_mid))

        return {
            'final_feat': x_deep,
            'shallow_feat': x_shallow,
            'xyz': xyz,
            'multi_scale_feats': [x_shallow, x_mid, x_deep]
        }


class MambaAdapter(BackboneAdapter):
    """Adapter for Mamba backbone."""

    def __init__(self):
        # Placeholder for Mamba backbone
        super().__init__(nn.Identity())
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

    def forward(self, x):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        x_shallow = F.relu(self.conv1(x))
        x_mid = F.relu(self.conv2(x_shallow))
        x_deep = F.relu(self.conv3(x_mid))

        return {
            'final_feat': x_deep,
            'shallow_feat': x_shallow,
            'xyz': xyz,
            'multi_scale_feats': [x_shallow, x_mid, x_deep]
        }


def get_backbone(args):
    """Get backbone with adapter based on backbone_name."""
    backbone_name = getattr(args, "backbone_name", "dgcnn")

    if backbone_name == "dgcnn":
        if args.use_high_dgcnn:
            backbone = DGCNN_semseg(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        else:
            backbone = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        return DGCNNAdapter(backbone)
    elif backbone_name == "ptv3":
        return PTv3Adapter()
    elif backbone_name == "mamba":
        return MambaAdapter()
    else:
        raise ValueError(f"Unknown backbone name: {backbone_name}")
