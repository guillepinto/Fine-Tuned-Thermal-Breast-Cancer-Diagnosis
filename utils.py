from make_dataset import make_loader, get_data
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SGD
import torch # 2.4.0
from networks import resnet, vit, xception
from torchvision.transforms import v2 # 0.19.0
# Pytorch metrics
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryPrecision
import wandb # 0.17.1

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_transforms(augmentation=False):

  # https://pytorch.org/vision/main/transforms.html#performance-considerations
  transforms_list = [
    v2.ToImage(),
    v2.Resize((224,224))
  ]

  if augmentation:
      # print("Efectivamente, voy a hacer transformaciones")
      transforms_list.append(v2.RandomHorizontalFlip())
      # transforms_list.append(v2.RandomVerticalFlip())
      transforms_list.append(v2.RandomRotation(degrees=45)) # Aplica una rotación aleatoria de hasta 45 grados.
      transforms_list.append(v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.5)) # Aplica un desenfoque gaussiano con una probabilidad de 0.5.
      transforms_list.append(v2.RandomApply([v2.RandomAffine(degrees=0, translate=(0.05, 0.05))], p=0.5)) # Aplica pequeñas traslaciones (hasta el 5% del tamaño de la imagen).

  transform = v2.Compose(transforms_list)

  return transform

def make_model(architecture:str, fine_tune:str = 'classifier', n_channels:int=1):
  if architecture=='xception':
      model = xception(n_channels=n_channels, num_classes=1, fine_tune=fine_tune)
  elif architecture=='resnet':
      model = resnet(n_channels=n_channels, num_classes=1, fine_tune=fine_tune)
  elif architecture=='vit':
      model = vit(n_channels=n_channels, num_classes=1, fine_tune=fine_tune)
  return model

def build_optimizer(network, optimizer, learning_rate):
  if optimizer == "sgd":
      # print("Efectivamente, voy a usar SGD")
      optimizer = SGD(network.parameters(),
                            lr=learning_rate, momentum=0.9)
  elif optimizer == "adam":
      # print("Efectivamente, voy a usar Adam")
      optimizer = Adam(network.parameters(),
                              lr=learning_rate)
  return optimizer

def make(config, fold=None):
    
    # Make transforms for data
    transform = make_transforms(augmentation=config.augmented)

    # Make the data
    train, test = get_data(transform=transform, slices=1, 
                            normalize=config.normalize,
                            fold=fold, n_channels=config.n_channels)
    
    input_size = train[0][0].size()

    train_loader = make_loader(train, batch_size=config.batch_size)
    # val_loader = make_loader(val, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = make_model(config.architecture, config.fine_tune).to(DEVICE)

    del input_size

    # Make the loss 
    criterion = BCEWithLogitsLoss()
    # Gradient optimization algorithms. AFTER moving the model to the GPU.
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate)

    # N-epochs to train
    epochs = config.epochs

    # Make metrics
    accuracy_fn = BinaryAccuracy().to(DEVICE)
    f1_score_fn = BinaryF1Score().to(DEVICE)
    recall_fn = BinaryRecall().to(DEVICE)
    precision_fn = BinaryPrecision().to(DEVICE)

    return model, train_loader, test_loader, criterion, optimizer, accuracy_fn, f1_score_fn, recall_fn, precision_fn, epochs

TEAM = 'ai-uis'
PROJECT = 'dmr-ir-fine-tuning'

# Function to save checkpoints
def save_checkpoint(net, optimizer, epoch, loss, best_loss):
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'best_loss': best_loss
    }
    torch.save(checkpoint, "checkpoint.pth")
    wandb.save("checkpoint.pth")

# Function to load the last saved checkpoint from wandb
def load_checkpoint(net, optimizer, run_id:str):
    try:
      checkpoint_path = wandb.restore('checkpoint.pth', run_path=f'{TEAM}/{PROJECT}/{run_id}').name # restore checkpoint from wandb
      checkpoint = torch.load(checkpoint_path)
      net.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      epoch = checkpoint['epoch']
      loss = checkpoint['loss']
      best_loss = checkpoint['best_loss']
      return epoch, loss, best_loss

    except Exception as e:
      print(f"Error loading checkpoint: {e}. Starting from scratch.")
      return 0, None, float('inf')  # If there is an error, start from scratch


# Function to load the model from a wandb run
def load_model_for_inference(model, run_id:str, device:str='cpu'):
  try:
    # Restore and load the the saved model file from specific run
    checkpoint_path = wandb.restore('checkpoint.pth', run_path=f'{TEAM}/{PROJECT}/{run_id}').name

    # Load the model status from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Model loaded successfully for inference")

    return model
  except Exception as e:
    print(f'Error loading checkpoint: {e}. Base model returned.')
    return model
