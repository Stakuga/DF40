"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, EigenGradCAM, AblationCAM, ScoreCAM, FEM, FinerCAM, XGradCAM, EigenCAM, LayerCAM, FullGrad, DeepFeatureFactorization, ShapleyCAM, KPCA_CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

on_2060 = False#"2060" in torch.cuda.get_device_name()
def init_seed(config):
    # if config['manualSeed'] is None:
    #     config['manualSeed'] = random.randint(1, 10000)
    config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=True, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring

def reshape_transform(tensor, height=16, width=16):
    # tensor shape: [B, num_tokens, hidden_dim]
    # ignore [CLS] token and reshape to image grid
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    # print("type is {}".format(type(tensor)))
    # print("tensor is {}".format(tensor))
    # print("shape is {}".format(tensor.shape))
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

    

def go_crazy(model, test_data_loaders):
    wrapped_model = GradCAMWrapper(model)
    wrapped_model.eval()
    target_layers = [wrapped_model.model.backbone.encoder.layers[-1].layer_norm1]  # Adjust this to the appropriate layer (CHANGE)
    #cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
    #cam = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
    #cam = HiResCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
    #cam = EigenGradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
    #cam = AblationCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform, ablation_layer=AblationLayerVit())
    cam = ScoreCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)

    save_dir = "cam_outputs"
    os.makedirs(save_dir, exist_ok=True)

    keys = test_data_loaders.keys()
    for key in keys:
        for i, data_dict in enumerate(test_data_loaders[key]):
            # print("data_dict is {}".format(data_dict.keys()))
            image_tensor = data_dict['image'].to(device)  # shape: (B, C, H, W)
            labels = data_dict['label']         # list of B strings

            image_paths = data_dict['path']

            # print("image tensor shape is {}".format(image_tensor.shape))
            # print("labels are {}".format(labels))

            for j in range(image_tensor.shape[0]):

                original_path = image_paths[j][0] if isinstance(image_paths[j], list) else image_paths[j]
                original_path = 'datasets/' + original_path 
                

                input_tensor = image_tensor[j].unsqueeze(0)
                label = labels[j].item()
                # Forward pass + GradCAM
                with torch.no_grad():
                    output = wrapped_model(input_tensor)
                    print("Model output shape:", output.shape)
                    print("Model output:", output)
                predicted_label = output.argmax(dim=1).item()
                print("Predicted label is {}".format(predicted_label))
                print("Ground-truth label is {}".format(label))
                #targets = [ClassifierOutputTarget(label)]  # Use predicted or ground-truth label
                targets = None
                # targets = [ScalarOutputTarget()]  # Use scalar output target
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
                #grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=False, eigen_smooth=False)[0]  # shape (H, W)
                print(grayscale_cam.min(), grayscale_cam.max(), grayscale_cam.mean())

                # Read in original image from original path
                true_img = cv2.imread(original_path)
                true_rgb = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB) / 255.0


                # Crop image from 1024x1024 to 985x985
                height, width = true_img.shape[:2]

                # Define target size
                target_size = 985

                # Calculate crop coordinates (center crop)
                start_x = (width - target_size) // 2
                start_y = (height - target_size) // 2
                end_x = start_x + target_size
                end_y = start_y + target_size

                # Crop the image and rgb
                true_img = true_img[start_y:end_y, start_x:end_x]
                true_rgb = true_rgb[start_y:end_y, start_x:end_x]

                # Resize 224×224 heatmap to match your 985×985 crop
                heatmap_985 = cv2.resize(grayscale_cam, (985, 985))

                visualization = show_cam_on_image(true_rgb, heatmap_985, use_rgb=True)

                # # Convert image to numpy format
                # rgb_img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
                # rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-5)

                # # get original image too
                # original_img = rgb_img.copy()
                # orig_img_uint8 = (original_img * 255).astype(np.uint8)

                # Save the 985×985 results
                orig_path = os.path.join(save_dir, f"img_{i}_{j}_GT_{label}_P_{predicted_label}_orig.png")
                cv2.imwrite(orig_path, cv2.cvtColor((true_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                out_path = os.path.join(save_dir, f"img_{i}_{j}_GT_{label}_P_{predicted_label}.png")
                cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

                # # save
                # orig_path = os.path.join(save_dir, f"img_{i}_{j}_GT_{label}_P_{predicted_label}_orig.png")
                # cv2.imwrite(orig_path, cv2.cvtColor(orig_img_uint8, cv2.COLOR_RGB2BGR))

                # # Overlay heatmap on image
                # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                # # Save image
                # out_path = os.path.join(save_dir, f"img_{i}_{j}_GT_{label}_P_{predicted_label}.png")
                # cv2.imwrite(out_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))


        
def go_crazy_classification_only(model, test_data_loaders):
    wrapped_model = GradCAMWrapper(model)
    wrapped_model.eval()
    target_layers = [wrapped_model.model.backbone.encoder.layers[-1].layer_norm1]  # Adjust this to the appropriate layer (CHANGE)
    cam = ScoreCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)

    save_dir = "cam_outputs"
    os.makedirs(save_dir, exist_ok=True)

    keys = test_data_loaders.keys()
    for key in keys:
        for i, data_dict in enumerate(test_data_loaders[key]):
            # print("data_dict is {}".format(data_dict.keys()))
            image_tensor = data_dict['image'].to(device)  # shape: (B, C, H, W)
            labels = data_dict['label']         # list of B strings

            image_paths = data_dict['path']

            # print("image tensor shape is {}".format(image_tensor.shape))
            # print("labels are {}".format(labels))

            for j in range(image_tensor.shape[0]):

                original_path = image_paths[j][0] if isinstance(image_paths[j], list) else image_paths[j]
                original_path = 'datasets/' + original_path 
                

                input_tensor = image_tensor[j].unsqueeze(0)
                label = labels[j].item()
                # Forward pass + GradCAM
                with torch.no_grad():
                    output = wrapped_model(input_tensor)
                    print("Model output shape:", output.shape)
                    print("Model output:", output)
                predicted_label = output.argmax(dim=1).item()
                print("Predicted label is {}".format(predicted_label))
                print("Ground-truth label is {}".format(label))

                # Read in original image from original path
                true_img = cv2.imread(original_path)
                true_rgb = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB) / 255.0


                # Crop image from 1024x1024 to 985x985
                height, width = true_img.shape[:2]

                # Define target size
                target_size = 985

                # Calculate crop coordinates (center crop)
                start_x = (width - target_size) // 2
                start_y = (height - target_size) // 2
                end_x = start_x + target_size
                end_y = start_y + target_size

                # Crop the image and rgb
                true_img = true_img[start_y:end_y, start_x:end_x]
                true_rgb = true_rgb[start_y:end_y, start_x:end_x]



                # Save the 985×985 results
                orig_path = os.path.join(save_dir, f"img_{i}_{j}_GT_{label}_P_{predicted_label}_orig.png")
                cv2.imwrite(orig_path, cv2.cvtColor((true_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))



def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if on_2060:
        config['lmdb_dir'] = r'I:\transform_2_lmdb'
        config['train_batchSize'] = 10
        config['workers'] = 0
    else:
        config['workers'] = 8
        config['lmdb_dir'] = r'/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/jikangcheng/data/LMDBs'
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        new_weights = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.', '')  # 删除module前缀
            if 'base_model.' in new_key:
                new_key = new_key.replace('base_model.', 'backbone.')
            if 'classifier.' in new_key:
                new_key = new_key.replace('classifier.', 'head.')
            new_weights[new_key] = value
        

        model.load_state_dict(new_weights, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    # best_metric = test_epoch(model, test_data_loaders)
    # print('===> Test Done!')

    # print('Starting GradCAM visualization...')
    # go_crazy(model, test_data_loaders)
    # print('GradCAM visualization done!')

    print('Starting general classification...')
    go_crazy_classification_only(model, test_data_loaders)
    print('General classification done!')

if __name__ == '__main__':
    main()
