from torchvision.models import resnet152, ResNet152_Weights # 0.19.0
from torchvision.models import vit_b_16, ViT_B_16_Weights # 0.19.0
from timm import create_model # 0.9.2
import torch.nn as nn # 2.4.0

def resnet(n_channels: int, num_classes: int, fine_tune: str = 'classifier'):
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)

    # Fine-tuning
    if fine_tune == 'classifier':
        # Congelar todas las capas excepto el clasificador
        for param in model.parameters():
            param.requires_grad = False
        # Descongelar el clasificador
        for param in model.fc.parameters():
            param.requires_grad = True

    elif fine_tune == 'classifier+conv1':
        # Modificar la primera capa convolucional para adaptarla a n_channels
        model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Congelar todas las capas excepto el clasificador y conv1
        for name, param in model.named_parameters():
            if 'conv1' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif fine_tune == 'full':
        # Modificar la primera capa convolucional para adaptarla a n_channels
        model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Descongelar todas las capas
        for param in model.parameters():
            param.requires_grad = True

    # Modificar el clasificador
    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    return model

def vit(n_channels: int, num_classes: int, fine_tune: str = 'classifier'):
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # Fine-tuning
    if fine_tune == 'classifier':
        # Congelar todas las capas excepto el clasificador
        for param in model.parameters():
            param.requires_grad = False
        # Descongelar el clasificador
        for param in model.heads[0].parameters():
            param.requires_grad = True

    elif fine_tune == 'classifier+conv1':
        # Modificar la proyección conv para adaptarla a n_channels
        model.conv_proj = nn.Conv2d(n_channels, 768, kernel_size=(16, 16), stride=(16, 16))
        # Congelar todas las capas excepto el clasificador y conv_proj
        for name, param in model.named_parameters():
            if 'conv_proj' in name or 'heads' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif fine_tune == 'full':
        # Modificar la proyección conv para adaptarla a n_channels
        model.conv_proj = nn.Conv2d(n_channels, 768, kernel_size=(16, 16), stride=(16, 16))
        # Descongelar todas las capas
        for param in model.parameters():
            param.requires_grad = True

    # Modificar el clasificador
    model.heads[0] = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    return model

def xception(n_channels: int, num_classes: int, fine_tune: str = 'classifier'):
    model = create_model('xception', pretrained=True, num_classes=num_classes)

    # Fine-tuning
    if fine_tune == 'classifier':
        # Congelar todas las capas excepto el clasificador
        for param in model.parameters():
            param.requires_grad = False
        # Descongelar el clasificador
        for param in model.fc.parameters():
            param.requires_grad = True

    elif fine_tune == 'classifier+conv1':
        # Modificar la primera capa convolucional para adaptarla a n_channels
        model.conv1 = nn.Conv2d(n_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        # Congelar todas las capas excepto el clasificador y conv1
        for name, param in model.named_parameters():
            if 'conv1' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif fine_tune == 'full':
        # Modificar la primera capa convolucional para adaptarla a n_channels
        model.conv1 = nn.Conv2d(n_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        # Descongelar todas las capas
        for param in model.parameters():
            param.requires_grad = True

    # Modificar el clasificador
    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    return model

  