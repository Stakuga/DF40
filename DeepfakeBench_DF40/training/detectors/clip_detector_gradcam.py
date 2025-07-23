'''
# author: Zhiyuan Yan (modified with Grad-CAM support)
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the CLIPDetector with Grad-CAM visualization support

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation
10. generate_gradcam: Generate Grad-CAM visualizations
11. save_gradcam_batch: Save Grad-CAM visualizations for a batch

Reference:
@inproceedings{rossler2019faceforensics++,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1--11},
  year={2019}
}
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union, Optional, Dict, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

# Import our Grad-CAM implementation
from .utils.gradcam import GradCAM, GradCAMPlusPlus

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='clip_gradcam')
class CLIPDetectorWithGradCAM(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)  # for CLIP-large-14
        # self.head = nn.Linear(768, 2) # for CLIP-base-16
        self.loss_func = self.build_loss(config)
        
        # Grad-CAM setup
        self.gradcam = None
        self.gradcam_enabled = config.get('gradcam_enabled', False)
        self.gradcam_layers = config.get('gradcam_layers', ['encoder.layers.23'])  # Last transformer layer
        self.gradcam_save_dir = config.get('gradcam_save_dir', './gradcam_visualizations')
        
        if self.gradcam_enabled:
            self._initialize_gradcam()
    
    def _initialize_gradcam(self):
        """Initialize Grad-CAM with appropriate target layers."""
        device = next(self.parameters()).device
        self.gradcam = GradCAM(
            model=self,
            target_layers=self.gradcam_layers,
            device=device
        )
        
        # Create save directory if it doesn't exist
        os.makedirs(self.gradcam_save_dir, exist_ok=True)
        logger.info(f"Grad-CAM initialized with layers: {self.gradcam_layers}")
        logger.info(f"Grad-CAM visualizations will be saved to: {self.gradcam_save_dir}")
    
    def build_backbone(self, config):
        # please download the ckpts from the below link:
        
        # use CLIP-base-16
        # _, backbone = get_clip_visual(model_name="openai/clip-vit-base-patch16")        
        
        # use CLIP-large-14
        _, backbone = get_clip_visual(model_name="openai/clip-vit-large-patch14")      
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict
    
    def generate_gradcam(self, 
                        data_dict: dict, 
                        target_class: Optional[int] = None,
                        layer_name: Optional[str] = None,
                        save_visualizations: bool = False,
                        filename_prefix: str = "gradcam") -> Dict[str, torch.Tensor]:
        """
        Generate Grad-CAM visualizations for the input.
        
        Args:
            data_dict: Input data dictionary containing 'image' key
            target_class: Target class for Grad-CAM. If None, uses predicted class.
            layer_name: Specific layer to generate CAM for. If None, uses all configured layers.
            save_visualizations: Whether to save visualizations to disk
            filename_prefix: Prefix for saved visualization files
            
        Returns:
            Dictionary mapping layer names to CAM tensors
        """
        if not self.gradcam_enabled or self.gradcam is None:
            logger.warning("Grad-CAM is not enabled. Please set gradcam_enabled=True in config.")
            return {}
        
        input_tensor = data_dict['image']
        
        # Generate CAM
        cams = self.gradcam.generate_cam(input_tensor, target_class, layer_name)
        
        if save_visualizations:
            self._save_gradcam_visualizations(
                input_tensor, cams, target_class, filename_prefix
            )
        
        return cams
    
    def _save_gradcam_visualizations(self, 
                                   input_tensor: torch.Tensor,
                                   cams: Dict[str, torch.Tensor],
                                   target_class: Optional[int] = None,
                                   filename_prefix: str = "gradcam"):
        """Save Grad-CAM visualizations to disk."""
        batch_size = input_tensor.size(0)
        
        for batch_idx in range(batch_size):
            # Prepare original image for visualization
            original_img = input_tensor[batch_idx]  # [C, H, W]
            
            # Denormalize if needed (assuming ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original_img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original_img.device)
            original_img_denorm = original_img * std + mean
            original_img_denorm = torch.clamp(original_img_denorm, 0, 1)
            
            for layer_name, cam in cams.items():
                # Take the specific batch sample
                cam_single = cam[batch_idx:batch_idx+1]  # Keep batch dimension
                
                # Resize CAM to match input image size
                input_size = original_img.shape[1:]  # [H, W]
                cam_resized = self.gradcam.resize_cam(cam_single, input_size)
                
                # Create visualization
                visualization = self.gradcam.visualize_cam(
                    original_img_denorm, cam_resized.squeeze(0)
                )
                
                # Save visualization
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename_prefix}_{layer_name}_batch{batch_idx}_{timestamp}.png"
                save_path = os.path.join(self.gradcam_save_dir, filename)
                
                # Get prediction info for title
                with torch.no_grad():
                    pred_dict = self.forward({'image': input_tensor[batch_idx:batch_idx+1]}, inference=True)
                    pred_class = pred_dict['cls'].argmax(dim=1).item()
                    confidence = torch.softmax(pred_dict['cls'], dim=1).max().item()
                
                title = f"Grad-CAM: {layer_name}\nPredicted: {pred_class} (conf: {confidence:.3f})"
                self.gradcam.save_visualization(visualization, save_path, title)
                
                logger.info(f"Saved Grad-CAM visualization: {save_path}")
    
    def save_gradcam_batch(self, 
                          data_dict: dict,
                          save_dir: Optional[str] = None,
                          filename_prefix: str = "batch_gradcam") -> Dict[str, List[np.ndarray]]:
        """
        Generate and save Grad-CAM visualizations for an entire batch.
        
        Args:
            data_dict: Batch data dictionary
            save_dir: Directory to save visualizations (if None, uses default)
            filename_prefix: Prefix for saved files
            
        Returns:
            Dictionary mapping layer names to lists of visualization arrays
        """
        if not self.gradcam_enabled:
            logger.warning("Grad-CAM is not enabled.")
            return {}
        
        if save_dir is None:
            save_dir = self.gradcam_save_dir
        
        # Generate CAMs
        cams = self.generate_gradcam(data_dict)
        
        # Process each sample in the batch
        input_tensor = data_dict['image']
        batch_size = input_tensor.size(0)
        
        visualizations = {}
        
        for layer_name, cam in cams.items():
            layer_visualizations = []
            
            for batch_idx in range(batch_size):
                # Prepare original image
                original_img = input_tensor[batch_idx]
                
                # Denormalize image
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original_img.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original_img.device)
                original_img_denorm = original_img * std + mean
                original_img_denorm = torch.clamp(original_img_denorm, 0, 1)
                
                # Get CAM for this sample
                cam_single = cam[batch_idx:batch_idx+1]
                input_size = original_img.shape[1:]
                cam_resized = self.gradcam.resize_cam(cam_single, input_size)
                
                # Create visualization
                visualization = self.gradcam.visualize_cam(
                    original_img_denorm, cam_resized.squeeze(0)
                )
                
                layer_visualizations.append(visualization)
                
                # Save individual visualization
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename_prefix}_{layer_name}_sample{batch_idx}_{timestamp}.png"
                save_path = os.path.join(save_dir, filename)
                
                # Get prediction for title
                with torch.no_grad():
                    sample_data = {'image': input_tensor[batch_idx:batch_idx+1]}
                    pred_dict = self.forward(sample_data, inference=True)
                    pred_class = pred_dict['cls'].argmax(dim=1).item()
                    confidence = torch.softmax(pred_dict['cls'], dim=1).max().item()
                
                title = f"Sample {batch_idx} - {layer_name}\nPred: {pred_class} (conf: {confidence:.3f})"
                self.gradcam.save_visualization(visualization, save_path, title)
            
            visualizations[layer_name] = layer_visualizations
        
        logger.info(f"Saved Grad-CAM visualizations for batch of size {batch_size} to {save_dir}")
        return visualizations
    
    def enable_gradcam(self, layers: Optional[List[str]] = None):
        """Enable Grad-CAM functionality."""
        if layers:
            self.gradcam_layers = layers
        self.gradcam_enabled = True
        self._initialize_gradcam()
    
    def disable_gradcam(self):
        """Disable Grad-CAM functionality."""
        self.gradcam_enabled = False
        if self.gradcam:
            self.gradcam.cleanup()
            self.gradcam = None
    
    def set_gradcam_layers(self, layers: List[str]):
        """Set target layers for Grad-CAM."""
        self.gradcam_layers = layers
        if self.gradcam_enabled:
            self.disable_gradcam()
            self.enable_gradcam()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.gradcam:
            self.gradcam.cleanup()


def get_clip_visual(model_name = "openai/clip-vit-base-patch16"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model

def get_vit_model(model_name = "google/vit-base-patch16-224-in21k"):
    #processor = AutoProcessor.from_pretrained(model_name)
    configuration = ViTConfig(
        image_size=224,
    )
    model = ViTModel.from_pretrained(model_name, config=configuration)
    return None, model
