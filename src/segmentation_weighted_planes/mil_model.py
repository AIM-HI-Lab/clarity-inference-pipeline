"""
MIL network + validation loop extracted for inference-only builds.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    convnext_tiny,
    resnet18,
    resnet34,
    resnet50,
    ConvNeXt_Tiny_Weights,
)

from segmentation_weighted_planes.data_loader_v5 import SWPDataset_V5
from segmentation_weighted_planes.projects import TrainingProject
from segmentation_weighted_planes.training.training_parameters import TrainingParameters

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class MILAttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.Tanh(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, feats):
        a = self.attn(feats).squeeze(-1)
        w = torch.softmax(a, dim=1)
        z = torch.sum(feats * w.unsqueeze(-1), 1)
        return z, w


class MILNet(nn.Module):
    def __init__(self, n_classes: int, pooling: str = "attn", topk: int = 8):
        super().__init__()
        self.pooling = pooling
        self.topk = topk

        base_model = getattr(TrainingParameters, "BASE_MODEL", "resnet50")
        if base_model == "resnet50":
            base = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif base_model == "resnet34":
            base = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif base_model == "resnet18":
            base = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif base_model == "convnext_tiny":
            base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        else:
            base = resnet50(weights=ResNet50_Weights.DEFAULT)

        if hasattr(base, "fc"):
            d = base.fc.in_features
            base.fc = nn.Identity()
        else:
            d = base.classifier[-1].in_features
            base.classifier = nn.Identity()

        self.encoder = base

        self.attn_pool = MILAttentionPool(d)
        self.head = nn.Linear(d, n_classes)

        feat_p = float(getattr(TrainingParameters, "FEAT_DROPOUT_P", 0.25))
        bag_p = float(getattr(TrainingParameters, "BAG_DROPOUT_P", 0.25))
        self.feat_drop = nn.Dropout(p=feat_p)
        self.bag_drop = nn.Dropout(p=bag_p)

        self.instance_drop_p = float(getattr(TrainingParameters, "INSTANCE_DROP_P", 0.20))
        self.min_keep = int(getattr(TrainingParameters, "MIN_KEEP_INSTANCES", 4))

        self.debug_attn = bool(getattr(TrainingParameters, "DEBUG_ATTN", False))

    def pool(self, feats: torch.Tensor):
        if self.pooling == "mean":
            return feats.mean(dim=1), None
        if self.pooling == "max":
            z, _ = feats.max(dim=1)
            return z, None
        if self.pooling == "topk":
            scores = torch.norm(feats, dim=-1)
            k = min(self.topk, feats.shape[1])
            inds = torch.topk(scores, k=k, dim=1).indices
            sel = torch.gather(
                feats, 1, inds.unsqueeze(-1).expand(-1, -1, feats.shape[-1])
            )
            return sel.mean(dim=1), None
        if self.pooling == "logsumexp":
            return torch.logsumexp(feats, dim=1), None
        return self.attn_pool(feats)

    def forward(self, bag_x: torch.Tensor, encoder_chunk_size: int = 32):
        B, K, C, H, W = bag_x.shape
        x = bag_x.view(B * K, C, H, W)

        feats_chunks = []
        for s in range(0, x.shape[0], encoder_chunk_size):
            feats_chunks.append(self.encoder(x[s : s + encoder_chunk_size]))
        f = torch.cat(feats_chunks, dim=0)

        D = f.shape[-1]
        feats = f.view(B, K, D)

        feats = self.feat_drop(feats)
        z, w = self.pool(feats)
        z = self.bag_drop(z)
        logits = self.head(z)

        if self.debug_attn and (w is not None) and self.training:
            if torch.rand(()) < 0.01:
                max_w = w.max(dim=1).values.mean().item()
                ent = (-w * (w.clamp_min(1e-8).log())).sum(dim=1).mean().item()
                print(f"[attn] mean max_w={max_w:.3f}  mean entropy={ent:.3f}")

        return logits, w


def pearson_coeff(labels, preds):
    mean_labels = np.mean(labels)
    mean_preds = np.mean(preds)
    numerator = np.sum((labels - mean_labels) * (preds - mean_preds))
    denominator = np.sqrt(
        np.sum((labels - mean_labels) ** 2) * np.sum((preds - mean_preds) ** 2)
    )
    return numerator / denominator


def get_n_class_metrics(labels, preds, pred_probs):
    labels = np.array(labels)
    preds = np.array(preds)

    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)

    auc = 0.0
    try:
        pred_probs_np = np.array(pred_probs)
        if pred_probs_np.ndim == 2 and pred_probs_np.shape[1] > 2:
            auc = roc_auc_score(labels, pred_probs_np, multi_class="ovr")
        elif pred_probs_np.ndim == 1:
            auc = roc_auc_score(labels, pred_probs_np)
    except ValueError:
        auc = 0.0

    conf_matrix = confusion_matrix(labels, preds)
    mcc = matthews_corrcoef(labels, preds)

    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp
    tn = np.sum(conf_matrix) - tp - fp - fn
    micro_prec = np.sum(tp) / (np.sum(tp) + np.sum(fp))
    micro_rec = np.sum(tp) / (np.sum(tp) + np.sum(fn))
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)

    tp_plus_fp = tp + fp
    precision_per_class = np.divide(
        tp, tp_plus_fp, out=np.zeros_like(tp, dtype=float), where=tp_plus_fp != 0
    )
    macro_prec = np.mean(precision_per_class)

    tp_plus_fn = tp + fn
    recall_per_class = np.divide(
        tp, tp_plus_fn, out=np.zeros_like(tp, dtype=float), where=tp_plus_fn != 0
    )
    macro_rec = np.mean(recall_per_class)

    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec)

    return {
        "tp": tp.tolist(),
        "tn": tn.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist(),
        "micro_prec": float(micro_prec),
        "micro_rec": float(micro_rec),
        "micro_f1": float(micro_f1),
        "macro_prec": float(macro_prec),
        "macro_rec": float(macro_rec),
        "macro_f1": float(macro_f1),
        "auc": float(auc),
        "conf_matrix": conf_matrix.tolist(),
        "mcc": float(mcc),
    }


def get_continuous_metrics(labels, preds, pred_probs):
    labels = np.array(labels)
    preds = np.array(preds)

    mse = np.mean((labels - preds) ** 2)
    rms = np.sqrt(mse)
    mae = np.mean(np.abs(labels - preds))
    pcc = pearson_coeff(labels, preds)

    return {"mse": float(mse), "rms": float(rms), "mae": float(mae), "pcc": float(pcc)}


def get_binary_metrics(labels, preds, pred_probs):
    labels = np.array(labels)
    preds = np.array(preds)

    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

    try:
        auc = roc_auc_score(labels, pred_probs)
    except ValueError:
        auc = 0.0

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "auc": float(auc),
    }


def validate_case_mil(net, dataset, case_idx, project_class):
    force_stack_mode = "center"

    y = dataset.data[case_idx]["label"]
    probs_accum = []

    n_bags = int(getattr(project_class, "bags_per_case_val", 4))

    net.eval()
    with torch.no_grad():
        for _ in range(n_bags):
            try:
                bag_x, _ = dataset.get_case_bag(case_idx=case_idx, augment=False)
            except TypeError:
                K = int(getattr(project_class, "bag_k", 32))
                bag_mix = getattr(project_class, "bag_mix", None) or {}
                try:
                    bag_x, _ = dataset.get_case_bag(
                        case_idx=case_idx,
                        k=K,
                        bag_mix=bag_mix,
                        augment=False,
                        use_seg=bool(getattr(project_class, "use_seg", True)),
                        stack_mode=force_stack_mode,
                    )
                except TypeError:
                    bag_x, _ = dataset.get_case_bag(
                        case_idx=case_idx,
                        k=K,
                        bag_mix=bag_mix,
                        augment=False,
                        use_seg=bool(getattr(project_class, "use_seg", True)),
                    )

            bag_x = torch.from_numpy(bag_x[None]).float().to(TrainingParameters.DEVICE)
            encoder_chunk = int(getattr(TrainingParameters, "ENCODER_CHUNK_SIZE", 32))
            use_amp = bool(getattr(TrainingParameters, "AMP", True)) and (
                TrainingParameters.DEVICE == "cuda"
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = net(bag_x, encoder_chunk_size=encoder_chunk)

            if project_class.n_classes == 1:
                probs_accum.append(float(logits.squeeze().item()))
            elif project_class.n_classes == 2:
                p1 = torch.sigmoid(logits[:, 1]).cpu().item()
                probs_accum.append(p1)
            else:
                probs_accum.append(torch.softmax(logits, 1).cpu().numpy()[0])

    if project_class.n_classes == 1:
        pred = float(np.mean(probs_accum))
        pred_probs = [pred]
    elif project_class.n_classes == 2:
        p1 = float(np.mean(probs_accum))
        pred = int(p1 >= 0.5)
        pred_probs = [1.0 - p1, p1]
    else:
        p = np.mean(np.stack(probs_accum, 0), 0)
        pred = int(np.argmax(p))
        pred_probs = [float(x) for x in p]

    return pred_probs, pred, y, dataset.data[case_idx]["case_id"]
