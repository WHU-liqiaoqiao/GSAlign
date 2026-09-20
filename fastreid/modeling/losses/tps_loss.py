# loss_head.py
import torch.nn as nn
import torch.nn.functional as  F
class MultiComponentLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mask_loss_fn = nn.MSELoss()
        self.tps_loss_weight = cfg.SOLVER.TPS_LOSS_WEIGHT
        self.mask_loss_weight = cfg.SOLVER.MASK_LOSS_WEIGHT

    def forward(self, outputs, targets, extra_info=None):
        """
        outputs: model 输出的 (global_feat, ...)
        targets: ground truth labels
        extra_info: dict 包含 masked_feat, mask, tps delta 信息等
        """
        global_feat = outputs[0].squeeze(-1).squeeze(-1)
        
        # Triplet Loss
        triplet_loss = self.tri_loss(global_feat, targets)

        # Mask Loss
        mask_loss = 0.
        if extra_info is not None and 'masked_feat' in extra_info:
            masked_feat = extra_info['masked_feat']  # (B, K, D)
            proto_feat = extra_info['proto_feat']    # (B, K, D)
            K = masked_feat.size(1)
            for i in range(K):
                mask_loss += self.mask_loss_fn(
                    F.normalize(masked_feat[:, i], dim=-1),
                    F.normalize(proto_feat[:, i], dim=-1)
                )
            mask_loss = mask_loss / K
        
        # TPS Loss
        tps_loss = 0.
        if extra_info is not None and 'rotation_angles' in extra_info and 'delta_list' in extra_info:
            for angle, delta in zip(extra_info['rotation_angles'], extra_info['delta_list']):
                tps_loss += angle.abs().mean() + delta.abs().mean()
                #tps_loss += 0 * angle.abs().mean() + delta.abs().mean()
            tps_loss = tps_loss / len(extra_info['rotation_angles'])

        total_loss = self.mask_loss_weight * mask_loss + \
                     self.tps_loss_weight * tps_loss

        return total_loss, {
            "mask_loss": mask_loss,
            "tps_loss": tps_loss
        }
