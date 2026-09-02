import os
import numpy as np
import cv2
import random
import yaml

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from google import genai
from google.genai import types

from detectors import DETECTOR

import argparse

from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./training/config/detector/clip.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str,
                    default='./training/df40_weights/train_on_fs/clip_large.pth',
                    help='path to the pre-trained model weights')
parser.add_argument('--xai', type=str, default=None,)
parser.add_argument('--image_path', type=str, default=None,)
args = parser.parse_args()

client = genai.Client(api_key='API_KEY') # replace 'API_KEY' with actual API key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def reshape_transform(tensor, height=16, width=16):
    # tensor shape: [B, num_tokens, hidden_dim]
    # ignore [CLS] token and reshape to image grid
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    result = tensor[:, 1:, :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class GradCAMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        data_dict = {'image': x}
        out = self.model(data_dict, inference=True)
        return out['cls']

class ScalarOutputTarget:
    def __call__(self, model_output):
        return model_output  # just return the scalar output directly

def go_classification_only(model, input_tensor, image_path):
    wrapped_model = GradCAMWrapper(model)
    wrapped_model.eval()

    save_dir = "cam_outputs"
    os.makedirs(save_dir, exist_ok=True)

    # input tensor is only one image, so add batch dimension
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Forward pass + GradCAM
    with torch.no_grad():
        output = wrapped_model(input_tensor)
        print("Model output shape:", output.shape)
        print("Model output:", output)
    predicted_label = output.argmax(dim=1).item()

    # Read in original image from original path
    true_img = cv2.imread(image_path)
    true_rgb = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB) / 255.0

    predicted_class = None
    if predicted_label == 0:
        predicted_class = "Real"
    else:
        predicted_class = "Fake"

    image_name = image_path.split('/')[-1].split('.')[0]

    print("Image is predicted as {}.".format(predicted_class))

    # Save the 985×985 results
    orig_path = os.path.join(save_dir, f"{image_name}_Predicted_{predicted_class}_orig.png")
    cv2.imwrite(orig_path, cv2.cvtColor((true_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print("wrote orig image to {}".format(orig_path))

def go_textual_explanation(model, input_tensor, image_path):
    wrapped_model = GradCAMWrapper(model)
    wrapped_model.eval()

    save_dir = "cam_outputs"
    os.makedirs(save_dir, exist_ok=True)

    # input tensor is only one image, so add batch dimension
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Forward pass + GradCAM
    with torch.no_grad():
        output = wrapped_model(input_tensor)
        print("Model output shape:", output.shape)
        print("Model output:", output)
    predicted_label = output.argmax(dim=1).item()

    # Read in original image from original path
    true_img = cv2.imread(image_path)
    true_rgb = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB) / 255.0

    predicted_class = None
    if predicted_label == 0:
        predicted_class = "Real"
    else:
        predicted_class = "Fake"

    image_name = image_path.split('/')[-1].split('.')[0]

    print("Image is predicted as {}.".format(predicted_class))

    # Save the 985×985 results
    orig_path = os.path.join(save_dir, f"{image_name}_Predicted_{predicted_class}_orig.png")
    cv2.imwrite(orig_path, cv2.cvtColor((true_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print("wrote orig image to {}".format(orig_path))

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
            types.Part(text=f"This image is potentially real or deepfake. It has been classified as '{predicted_class.lower()}' by some deepfake detection model. Your role is a textual explanation generator. Briefly explain the classification.")
        ]
    )

    try:
        text = response.candidates[0].content.parts[0].text
        print(text)
    except Exception as e:  # catch anything
        print("Failed to get content from response:", e)

def go_visual_explanation(model, input_tensor, image_path):
    wrapped_model = GradCAMWrapper(model)
    wrapped_model.eval()
    target_layers = [wrapped_model.model.backbone.encoder.layers[-1].layer_norm1]
    cam = ScoreCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)

    save_dir = "cam_outputs"
    os.makedirs(save_dir, exist_ok=True)

    # input tensor is only one image, so add batch dimension
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Forward pass + GradCAM
    with torch.no_grad():
        output = wrapped_model(input_tensor)
        print("Model output shape:", output.shape)
        print("Model output:", output)
    predicted_label = output.argmax(dim=1).item()
    print("Predicted label is {}".format(predicted_label))

    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    print(grayscale_cam.min(), grayscale_cam.max(), grayscale_cam.mean())

    # Read in original image from original path
    true_img = cv2.imread(image_path)
    true_rgb = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB) / 255.0

    # Crop image from 1024x1024 to 985x985
    height, width = true_img.shape[:2]

    # Resize 224×224 heatmap to match original input image resolution
    heatmap_985 = cv2.resize(grayscale_cam, (height, width))

    visualization = show_cam_on_image(true_rgb, heatmap_985, use_rgb=True)

    predicted_class = None
    if predicted_label == 0:
        predicted_class = "Real"
    else:
        predicted_class = "Fake"

    image_name = image_path.split('/')[-1].split('.')[0]

    # Save the 985×985 results
    orig_path = os.path.join(save_dir, f"{image_name}_Predicted_{predicted_class}_orig.png")
    cv2.imwrite(orig_path, cv2.cvtColor((true_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print("wrote orig image to {}".format(orig_path))

    out_path = os.path.join(save_dir, f"{image_name}_Predicted_{predicted_class}.png")
    cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    print("wrote cam image to {}".format(out_path))



def go_textual_and_visual_explanation(model, input_tensor, image_path):
    wrapped_model = GradCAMWrapper(model)
    wrapped_model.eval()
    target_layers = [wrapped_model.model.backbone.encoder.layers[-1].layer_norm1]
    cam = ScoreCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)

    save_dir = "cam_outputs"
    os.makedirs(save_dir, exist_ok=True)

    # input tensor is only one image, so add batch dimension
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Forward pass + GradCAM
    with torch.no_grad():
        output = wrapped_model(input_tensor)
        print("Model output shape:", output.shape)
        print("Model output:", output)
    predicted_label = output.argmax(dim=1).item()
    print("Predicted label is {}".format(predicted_label))

    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    print(grayscale_cam.min(), grayscale_cam.max(), grayscale_cam.mean())

    # Read in original image from original path
    true_img = cv2.imread(image_path)
    true_rgb = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB) / 255.0

    # Crop image from 1024x1024 to 985x985
    height, width = true_img.shape[:2]

    # Resize 224×224 heatmap to match original input image resolution
    heatmap_985 = cv2.resize(grayscale_cam, (height, width))

    visualization = show_cam_on_image(true_rgb, heatmap_985, use_rgb=True)

    predicted_class = None
    if predicted_label == 0:
        predicted_class = "Real"
    else:
        predicted_class = "Fake"

    image_name = image_path.split('/')[-1].split('.')[0]

    # Save the 985×985 results
    orig_path = os.path.join(save_dir, f"{image_name}_Predicted_{predicted_class}_orig.png")
    cv2.imwrite(orig_path, cv2.cvtColor((true_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print("wrote orig image to {}".format(orig_path))

    out_path = os.path.join(save_dir, f"{image_name}_Predicted_{predicted_class}.png")
    cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    print("wrote cam image to {}".format(out_path))



    # stitch original image and cam image directly side by side with original on the left
    stitched_image = np.concatenate((cv2.cvtColor((true_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)), axis=1)
    stitched_out_path = os.path.join(save_dir, f"{image_name}_Predicted_{predicted_class}_stitched.png")
    cv2.imwrite(stitched_out_path, stitched_image)


    with open(stitched_out_path, 'rb') as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
            types.Part(text=f"This image input is composed of two images stitched together. The left image is potentially real or deepfake. It has been classified as '{predicted_class.lower()}' by some deepfake detection model. The right image shows a corresponding heatmap generated for this classification. Your role is a textual explanation generator. Briefly explain the classification.")
        ]
    )

    try:
        text = response.candidates[0].content.parts[0].text
        print(text)
    except Exception as e:  # catch anything
        print("Failed to get content from response:", e)


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    config['workers'] = 8

    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path

    # specify xAI
    xai_technique = None
    if args.xai:
        xai_technique = args.xai

    # specify image path
    image_path = None
    if args.image_path:
        image_path = args.image_path
    else:
        print("Please provide an input image path using --image_path argument.")
        return

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        new_weights = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.', '')  # remove the 'module.' prefix
            if 'base_model.' in new_key:
                new_key = new_key.replace('base_model.', 'backbone.')
            if 'classifier.' in new_key:
                new_key = new_key.replace('classifier.', 'head.')
            new_weights[new_key] = value

        model.load_state_dict(new_weights, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    # load input image
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(f"Error loading image at path {image_path}: {e}")
        return
    image = np.array(image)

    # turn input image to tensor and normalize
    image_tensor = transforms.ToTensor()(image)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)
    image_tensor = normalize(image_tensor)

    if xai_technique == 'textual':
        print('Starting textual explanation generation...')
        go_textual_explanation(model, image_tensor, image_path)
        print('Textual explanation generation done!')

    elif xai_technique == 'visual':
        print('Starting visual explanation generation...')
        go_visual_explanation(model, image_tensor, image_path)
        print('Visual explanation generation done!')

    elif xai_technique == 'multimodal':
        print('Starting textual and visual explanation generation...')
        go_textual_and_visual_explanation(model, image_tensor, image_path)
        print('Textual and visual explanation generation done!')

    else:
        print('No valid xAI technique specified, classifying image only.')
        go_classification_only(model, image_tensor, image_path)
        print('Classification done!')

if __name__ == '__main__':
    main()
