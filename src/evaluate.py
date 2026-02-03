import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device=device):

    """ Performs a testing with model trying to learn on data_loader"""

    test_loss, test_accuracy = 0, 0
    # Put the model in eval mode
    model.eval()
    # Turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data on target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y)

            # 3. Calculate accuracy
            test_accuracy += accuracy_fn(y_true = y,
                                    y_pred = test_pred.argmax(dim = 1))

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)

        # Print out what`s happening
        print(f'Test loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.2f}% \n')

        return test_loss, test_accuracy

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: str = None,
               return_preds: bool = False):
    """
    Evaluate model on data_loader.
    Returns a dict with average loss and accuracy.
    If return_preds=True, also return concatenated preds and targets (CPU tensors).
    """
    if device is None:
        # infer device from model parameters (works if model already moved to device)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    loss_total = 0.0
    acc_total = 0.0
    n_batches = 0

    preds_list = []
    targets_list = []

    with torch.inference_mode():
        for X, y in (data_loader):
            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            # handle binary logits shape or multiclass logits
            # assume loss_fn expects (logits, targets) (CrossEntropy, BCEWithLogits, etc.)
            loss = loss_fn(logits, y)
            # predictions: if multiclass -> argmax; if binary logits -> round(sigmoid)
            if logits.dim() > 1 and logits.shape[1] > 1:
                batch_preds = logits.argmax(dim=1)
            else:
                batch_preds = torch.round(torch.sigmoid(logits)).squeeze().to(torch.long)

            # accuracy_fn should accept (y_true, y_pred) and return a scalar (float or tensor)
            batch_acc = accuracy_fn(y_true=y, y_pred=batch_preds)

            loss_total += float(loss.item())
            # ensure numeric
            acc_total += float(batch_acc)
            n_batches += 1

            if return_preds:
                preds_list.append(batch_preds.cpu())
                targets_list.append(y.cpu())

    avg_loss = loss_total / max(1, n_batches)
    avg_acc = acc_total / max(1, n_batches)

    result = {
        "model_name": model.__class__.__name__,
        "model_loss": avg_loss,
        "model_accuracy": avg_acc
    }

    if return_preds:
        result["preds"] = torch.cat(preds_list)
        result["targets"] = torch.cat(targets_list)

    return result