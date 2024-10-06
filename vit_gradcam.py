import argparse
import cv2
import numpy as np
import torch
from networks import vit
from utils import get_data, make_transforms

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    
    model = vit(n_channels=1, num_classes=1, fine_tune='full')  # Asume que defines 'resnet' en tu archivo 'networks.py'
    """ Good Network """
    # checkpoint = torch.load('vit_32_full_00001/h7knv1x1_checkpoint.pth', map_location='cpu', weights_only=True)
    """ Bad Network """
    checkpoint = torch.load('checkpoints/bad_models/j3hibluq_bad_vit_16_full_0001.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = True

    # model = torch.hub.load('facebookresearch/deit:main',
    #                        'deit_tiny_patch16_224', pretrained=True)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    # target_layers = [model.blocks[-1].norm1]
    target_layers = [model.encoder.layers[-1].ln_1] # para mi ViT

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform)

    # rgb_img = cv2.imread(args.image_path, 1, )[:, :, ::-1]
    # rgb_img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    idx = 12
    data = get_data(transform=make_transforms(False), normalize=True, slices=10, fold=2)
    rgb_img = data[1][idx][0].numpy()

    # print(rgb_img.shape)
    # rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / np.max(rgb_img)
    rgb_img = np.float32(np.transpose(rgb_img, (1,2,0)))


    # input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
    #                                 std=[0.5, 0.5, 0.5])
    # input_tensor = preprocess_image(rgb_img, mean=[0.5],
    #                                 std=[0.5])
    input_tensor = data[1][idx][0].unsqueeze(0)

    # rgb_img = np.float32(np.expand_dims(rgb_img, axis=2))
    # rgb_img = np.float32(np.repeat(rgb_img, 3, axis=2))

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 2

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
