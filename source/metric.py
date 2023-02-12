import torch
import torchmetrics
import config


def decode(x: torch.Tensor) -> list[str]:
    names = []

    for i in range(len(x)):
        name = ""
        indices = torch.argmax(x[i], dim=1)

        for idx in indices:
            name += config.IDX_TO_SYMBOL[idx.item()]

        names.append(name)

    return names


def character_error_rate(
    predictions: torch.Tensor, labels: torch.Tensor
) -> float:
    y_pred = decode(predictions)
    y_true = decode(labels)
    metric = torchmetrics.CharErrorRate()
    return metric(y_pred, y_true)
