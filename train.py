from utils import load_checkpoint, save_checkpoint
import torch # 2.4.0
import torch.nn as nn
from tqdm import tqdm # 4.66.4
import wandb # 0.17.1
from test import test

# def train_log(loss, accuracy, step, current):
#     """ Log the metrics for the current batch into wandb

#     Args:
#         loss: the value of the loss at current batch
#         accuracy: the value of the accuracy at current batch
#         step: actual step
#         current: actual batch
#     """

#     # Where the magic happens
#     wandb.log({"step":step, "train_loss": loss, "train_accuracy": accuracy})
#     print(f"train loss: {loss:.3f} accuracy: {accuracy:.3f} [after {current} batches]")

def train(model: nn.Module, train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          criterion,
          optimizer,
          accuracy_fn,
          f1_score_fn,
          recall_fn,
          precision_fn,
          epochs: int,
          device: str,
          run_id: str=None):
    
    """
    Train the given model using the specified data loader, criterion, optimizer, and metric function.

    Parameters:
    model (torch.nn.Module): The neural network model to be trained.
    loader (torch.utils.data.DataLoader): The data loader providing training batches.
    criterion (torch.nn.Module): The loss function used to compute the loss.
    optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
    metric_fn (callable): The function used to compute the training metric (e.g., accuracy).
    
    Notes:
    - This function sets the model to training mode and iterates over the data loader.
    - The function computes the loss and accuracy for each batch, updates the model parameters,
      and logs metrics using `train_log` every `n_prints` batches.
    - The step count is incremented after each logging operation.
    """

    # If resume is specified, loads the last checkpoint.
    start_epoch = 0
    best_loss = float('inf')
    patience = 7
    if run_id is not None:
      start_epoch, _, best_loss = load_checkpoint(model, optimizer, run_id)
      print(f"Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")

    # 4 means that I am going to make 4 logs of the metrics when training
    # n_prints = int(len(train_loader)/4)

    # loop over the dataset multiple times
    for epoch in range(start_epoch, start_epoch+epochs):
        running_loss = 0.0
        running_acc = 0.0
        # Loop through dataloader with tqdm for batch progress
        batch_iterator = tqdm(enumerate(train_loader, 0), desc=f"Epoch {epoch+1}", leave=False, total=len(train_loader))
        for batch_idx, data in batch_iterator:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            train_loss = criterion(outputs, labels.unsqueeze(1).float())
            train_accuracy = accuracy_fn(outputs, labels.unsqueeze(1).float())

            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()
            running_acc += train_accuracy

            # Update the description of the progress bar with the running loss
            batch_iterator.set_postfix(loss=running_loss / (batch_idx + 1))

        # and validate its performance per epoch
        test_loss, test_accuracy, test_f1, test_recall, test_precision = test(model, test_loader, criterion, accuracy_fn, f1_score_fn, recall_fn, precision_fn, epoch)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        
        print(f'Epoch {epoch+1}/{start_epoch+epochs} - train_loss: {epoch_loss:.3f} train_accuracy {epoch_acc:.3f}')

        # Log to wandb 
        wandb.log({"epoch": epoch+1, "train_loss": epoch_loss, "train_accuracy": epoch_acc})
        # print(f"train loss: {epoch_loss:.3f} accuracy: {accuracy:.3f} [after {} batches]")

        # Save the best model and early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience = 7
            save_checkpoint(model, optimizer, epoch+1, test_loss, best_loss)
        else:
          patience -= 1
          if patience == 0:
              break
          
    print('Finished Training')
    return test_accuracy, test_f1, test_recall, test_precision
