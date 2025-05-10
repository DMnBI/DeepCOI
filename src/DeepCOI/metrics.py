import torch
from torchmetrics import Accuracy, AUROC


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
	num_labels = preds.shape[1]
	acc_fn = Accuracy(task='multilabel', num_labels=num_labels).to(preds.device)
	acc = acc_fn(preds, labels.long())

	return acc


def auroc(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
	num_labels = preds.shape[1]
	auroc_fn = AUROC(task="multilabel", num_labels=num_labels).to(preds.device)
	auc = auroc_fn(preds, labels.long())

	return auc
