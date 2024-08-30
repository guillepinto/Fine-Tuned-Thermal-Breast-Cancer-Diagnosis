import wandb
import torch
from utils import DEVICE

def test(model, test_loader, loss_fn, accuracy_fn, f1_score_fn, recall_fn, precision_fn, epoch):
    """ 
    Evaluate the model on the test dataset and log the performance metrics.

    Parameters:
    model (torch.nn.Module): The model to be evaluated.
    test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    accuracy_fn (function): Function to compute accuracy.
    f1_score_fn (function): Function to compute F1 score.
    recall_fn (function): Function to compute recall.
    precision_fn (function): Function to compute precision.
    """
    model.eval()

    # Run the model on some test examples
    num_batches = len(test_loader)
    test_loss, test_accuracy, test_f1, test_recall, test_precision = 0, 0, 0, 0, 0

    # Disable gradient calculation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            test_loss += loss_fn(outputs, labels.unsqueeze(1).float()).item()
            test_accuracy += accuracy_fn(outputs, labels.unsqueeze(1).float())
            test_f1 += f1_score_fn(outputs, labels.unsqueeze(1).float())
            test_recall += recall_fn(outputs, labels.unsqueeze(1).float())
            test_precision += precision_fn(outputs, labels.unsqueeze(1).float())

        # Average the metrics over all batches
        test_loss /= num_batches
        test_accuracy /= num_batches
        test_f1 /= num_batches
        test_recall /= num_batches
        test_precision /= num_batches

        wandb.log({"epoch": epoch+1, "test_accuracy": test_accuracy, "test_f1": test_f1,
                    "test_recall": test_recall, "test_precision": test_precision})
        print(f"test loss: {test_loss:.3f} accuracy: {test_accuracy:.3f} recall: {test_recall:.3f} precision: {test_precision:.3f} f1: {test_f1:.3f}")

    return test_loss, test_accuracy, test_f1, test_recall, test_precision