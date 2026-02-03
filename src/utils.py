import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Plot Predictions
def plot_predictions(model, dataloader, class_names, device=None, n_images=9, denormalize=None, save_path=None):
    """
    Show n_images predictions from first batch in dataloader.
    denormalize: function to reverse transforms (image_tensor -> image_numpy), optional.
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    model.eval()
    with torch.inference_mode():
        images, labels = next(iter(dataloader))
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if logits.dim() > 1 and logits.shape[1] > 1:
            preds = logits.argmax(dim=1)
        else:
            preds = torch.round(torch.sigmoid(logits)).squeeze().to(torch.long)

    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    n = min(n_images, images.shape[0])
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))

    plt.figure(figsize=(ncols * 3, nrows * 3))
    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)
        img = images[i].squeeze().numpy()
        if denormalize:
            img = denormalize(img)
        plt.imshow(img, cmap="gray")
        pred_label = class_names[int(preds[i])]
        true_label = class_names[int(labels[i])]
        title = f"Pred: {pred_label}\nTrue: {true_label}"
        color = "green" if pred_label == true_label else "red"
        plt.title(title, color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()


# Plot Confusion Matrix
def plot_confusion_matrix(preds, targets, class_names, figsize=(10,8), normalize=False, save_path=None):
    """
    preds, targets: 1D numpy/torch arrays (or tensors)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    cm = confusion_matrix(targets, preds, labels=range(len(class_names)))
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion Matrix saved to {save_path}")
    plt.show()