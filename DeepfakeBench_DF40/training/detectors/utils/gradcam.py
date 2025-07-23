"""
Grad-CAM implementation for deepfake detection models
Author: GitHub Copilot Assistant  
Description: Gradient-based Class Activation Mapping (Grad-CAM) implementation
             for visualizing what regions the model focuses on during prediction.

Reference:
@article{selvaraju2017grad,
  title={Grad-CAM: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  journal={Proceedings of the IEEE international conference on computer vision},
  pages={618--626},
  year={2017}
}
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os


class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    """
    
    def __init__(self, model: torch.nn.Module, target_layers: List[str], device: str = 'cuda'):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The PyTorch model
            target_layers: List of layer names to compute Grad-CAM for
            device: Device to run computation on
        """
        self.model = model
        self.target_layers = target_layers
        self.device = device
        
        self.gradients = {}
        self.activations = {}
        self.handles = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture gradients and activations."""
        
        def save_gradient(name):
            def hook(grad):
                self.gradients[name] = grad
            return hook
        
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook
        
        # Register hooks for target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                # Forward hook to save activations
                handle_forward = module.register_forward_hook(save_activation(name))
                self.handles.append(handle_forward)
                
                # Backward hook to save gradients
                handle_backward = module.register_full_backward_hook(
                    lambda module, grad_input, grad_output, name=name: 
                    save_gradient(name)(grad_output[0])
                )
                self.handles.append(handle_backward)
    
    def generate_cam(self, 
                     input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None,
                     layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Generate Grad-CAM heatmaps.
        
        Args:
            input_tensor: Input tensor to the model [B, C, H, W]
            target_class: Target class index for Grad-CAM. If None, uses predicted class.
            layer_name: Specific layer to generate CAM for. If None, generates for all target layers.
            
        Returns:
            Dictionary mapping layer names to CAM tensors
        """
        self.model.eval()
        
        # Forward pass
        input_tensor.requires_grad_()
        data_dict = {'image': input_tensor}
        
        # Get model prediction
        pred_dict = self.model(data_dict, inference=True)
        logits = pred_dict['cls']
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * input_tensor.size(0), device=self.device)
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits.gather(1, target_class.unsqueeze(1)).squeeze()
        class_score.backward(torch.ones_like(class_score), retain_graph=True)
        
        # Generate CAMs
        cams = {}
        layers_to_process = [layer_name] if layer_name else self.target_layers
        
        for layer in layers_to_process:
            if layer in self.gradients and layer in self.activations:
                gradients = self.gradients[layer]  # [B, C, H, W]
                activations = self.activations[layer]  # [B, C, H, W]
                
                # Global average pooling of gradients
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
                
                # Weighted combination of activation maps
                cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [B, 1, H, W]
                
                # Apply ReLU and normalize
                cam = F.relu(cam)
                cam = self._normalize_cam(cam)
                
                cams[layer] = cam
        
        return cams
    
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """
        Normalize CAM to [0, 1] range.
        
        Args:
            cam: CAM tensor [B, 1, H, W]
            
        Returns:
            Normalized CAM tensor
        """
        B, C, H, W = cam.shape
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        
        # Avoid division by zero
        cam_range = cam_max - cam_min
        cam_range = torch.where(cam_range == 0, torch.ones_like(cam_range), cam_range)
        
        cam_normalized = (cam - cam_min) / cam_range
        return cam_normalized
    
    def resize_cam(self, cam: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Resize CAM to target size.
        
        Args:
            cam: CAM tensor [B, 1, H, W]
            target_size: Target size (height, width)
            
        Returns:
            Resized CAM tensor
        """
        return F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
    
    def visualize_cam(self, 
                      image: Union[torch.Tensor, np.ndarray], 
                      cam: torch.Tensor,
                      alpha: float = 0.4,
                      colormap: str = 'jet') -> np.ndarray:
        """
        Visualize CAM overlaid on original image.
        
        Args:
            image: Original image tensor [C, H, W] or numpy array [H, W, C]
            cam: CAM tensor [1, H, W] or [H, W]
            alpha: Blending factor for overlay
            colormap: Matplotlib colormap name
            
        Returns:
            Visualization as numpy array [H, W, C]
        """
        # Convert inputs to numpy
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # [C, H, W]
                image = image.permute(1, 2, 0).cpu().numpy()
            else:  # [H, W, C]
                image = image.cpu().numpy()
        
        if isinstance(cam, torch.Tensor):
            if cam.dim() == 3:  # [1, H, W]
                cam = cam.squeeze(0).cpu().numpy()
            else:  # [H, W]
                cam = cam.cpu().numpy()
        
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Ensure image is in [0, 1] range
        image = np.clip(image, 0, 1)
        
        # Apply colormap to CAM
        cmap = plt.get_cmap(colormap)
        cam_colored = cmap(cam)[:, :, :3]  # Remove alpha channel
        
        # Blend image and CAM
        visualization = alpha * cam_colored + (1 - alpha) * image
        visualization = np.clip(visualization, 0, 1)
        
        return (visualization * 255).astype(np.uint8)
    
    def save_visualization(self, 
                          visualization: np.ndarray, 
                          save_path: str,
                          title: Optional[str] = None):
        """
        Save visualization to file.
        
        Args:
            visualization: Visualization array [H, W, C]
            save_path: Path to save the visualization
            title: Optional title for the plot
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(visualization)
        plt.axis('off')
        if title:
            plt.title(title, fontsize=16)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def generate_multi_layer_visualization(self,
                                         input_tensor: torch.Tensor,
                                         original_image: Union[torch.Tensor, np.ndarray],
                                         target_class: Optional[int] = None,
                                         save_dir: Optional[str] = None,
                                         filename_prefix: str = "gradcam") -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM visualizations for all target layers.
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            original_image: Original image for visualization
            target_class: Target class for Grad-CAM
            save_dir: Directory to save visualizations
            filename_prefix: Prefix for saved files
            
        Returns:
            Dictionary mapping layer names to visualization arrays
        """
        # Generate CAMs
        cams = self.generate_cam(input_tensor, target_class)
        
        visualizations = {}
        
        # Get target size from original image
        if isinstance(original_image, torch.Tensor):
            if original_image.dim() == 3:  # [C, H, W]
                target_size = original_image.shape[1:]
            else:  # [B, C, H, W]
                target_size = original_image.shape[2:]
        else:  # numpy array [H, W, C]
            target_size = original_image.shape[:2]
        
        for layer_name, cam in cams.items():
            # Take first sample if batch
            if cam.dim() == 4:
                cam_single = cam[0]  # [1, H, W]
            else:
                cam_single = cam
            
            # Resize CAM to match original image size
            cam_resized = self.resize_cam(cam_single.unsqueeze(0), target_size).squeeze(0)
            
            # Create visualization
            if isinstance(original_image, torch.Tensor) and original_image.dim() == 4:
                img_single = original_image[0]  # Take first sample
            else:
                img_single = original_image
                
            visualization = self.visualize_cam(img_single, cam_resized)
            visualizations[layer_name] = visualization
            
            # Save if directory provided
            if save_dir:
                save_path = os.path.join(save_dir, f"{filename_prefix}_{layer_name}.png")
                pred_class = target_class if target_class is not None else "predicted"
                title = f"Grad-CAM: {layer_name} (Class: {pred_class})"
                self.save_visualization(visualization, save_path, title)
        
        return visualizations
    
    def cleanup(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.gradients.clear()
        self.activations.clear()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation with improved localization.
    
    Reference:
    @article{chattopadhay2018grad,
      title={Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks},
      author={Chattopadhay, Aditya and Sarkar, Anirban and Howlader, Prantik and Balasubramanian, Vineeth N},
      journal={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      pages={839--847},
      year={2018}
    }
    """
    
    def generate_cam(self, 
                     input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None,
                     layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Generate Grad-CAM++ heatmaps with improved weighting.
        """
        self.model.eval()
        
        # Forward pass
        input_tensor.requires_grad_()
        data_dict = {'image': input_tensor}
        
        pred_dict = self.model(data_dict, inference=True)
        logits = pred_dict['cls']
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * input_tensor.size(0), device=self.device)
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits.gather(1, target_class.unsqueeze(1)).squeeze()
        class_score.backward(torch.ones_like(class_score), retain_graph=True)
        
        # Generate CAMs with Grad-CAM++ weighting
        cams = {}
        layers_to_process = [layer_name] if layer_name else self.target_layers
        
        for layer in layers_to_process:
            if layer in self.gradients and layer in self.activations:
                gradients = self.gradients[layer]  # [B, C, H, W]
                activations = self.activations[layer]  # [B, C, H, W]
                
                # Grad-CAM++ weighting
                alpha_num = gradients.pow(2)
                alpha_denom = 2.0 * gradients.pow(2) + \
                             torch.sum(activations * gradients.pow(3), dim=(2, 3), keepdim=True)
                alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
                alpha = alpha_num / alpha_denom
                
                weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
                
                # Weighted combination
                cam = torch.sum(weights * activations, dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = self._normalize_cam(cam)
                
                cams[layer] = cam
        
        return cams
