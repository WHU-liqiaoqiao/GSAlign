# loss_head.py

import torch
import torch.nn.functional as F

def mask_loss(outputs, targets, alpha=0.5):
    """
    Args:
        outputs: model forward dict (带 "distill_features")
        targets: ground-truth labels
        alpha:   distillation loss 权重
    Returns:
        dict of losses
    """

    cls_outputs = outputs["cls_outputs"]
    pred_class_logits = outputs["pred_class_logits"]
    features = outputs["features"]
    distill_features = outputs["distill_features"]

    # 标准分类loss（CrossEntropy）
    loss_cls = F.cross_entropy(cls_outputs, targets)

    # Self-distillation loss（feature-level的一致性）
    # 可以用 cosine loss or MSE loss
    loss_distill = 1 - F.cosine_similarity(features, distill_features, dim=-1).mean()

    total_loss = loss_cls + alpha * loss_distill

    return {
        "loss_cls": loss_cls,
        "loss_distill": loss_distill * alpha,
        "loss_total": total_loss
    }
