"""
Grad-CAM Demo Script for CLIP Detector
Author: GitHub Copilot Assistant

This script demonstrates how to use the Grad-CAM functionality with the CLIP detector
for deepfake detection visualization.

Usage:
    python gradcam_demo.py --config path/to/config.yaml --input path/to/image.jpg --output ./gradcam_output
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.clip_detector_gradcam import CLIPDetectorWithGradCAM
from detectors.utils.gradcam import GradCAM, GradCAMPlusPlus


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path, image_size=224):
    """Load and preprocess image."""
    # Define transforms (assuming ImageNet normalization for CLIP)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_image


def create_sample_config():
    """Create a sample configuration for the CLIP detector with Grad-CAM."""
    config = {
        'loss_func': 'cross_entropy_loss',
        'gradcam_enabled': True,
        'gradcam_layers': [
            'encoder.layers.23',  # Last transformer layer
            'encoder.layers.20',  # Earlier layer for comparison  
            'encoder.layers.15',  # Even earlier layer
        ],
        'gradcam_save_dir': './gradcam_visualizations'
    }
    return config


def demonstrate_gradcam(image_path, output_dir, config_path=None):
    """
    Demonstrate Grad-CAM functionality on a single image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        config_path: Path to config file (optional)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        print("Using sample configuration...")
        config = create_sample_config()
        config['gradcam_save_dir'] = output_dir
    
    # Initialize model
    print("Initializing CLIP detector with Grad-CAM...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPDetectorWithGradCAM(config).to(device)
    model.eval()
    
    # Load image
    print(f"Loading image: {image_path}")
    image_tensor, original_image = load_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Prepare data dictionary
    data_dict = {'image': image_tensor}
    
    # Forward pass to get prediction
    print("Running forward pass...")
    with torch.no_grad():
        pred_dict = model(data_dict, inference=True)
        predicted_class = pred_dict['cls'].argmax(dim=1).item()
        confidence = torch.softmax(pred_dict['cls'], dim=1).max().item()
        probabilities = torch.softmax(pred_dict['cls'], dim=1)[0]
    
    print(f"Prediction: Class {predicted_class} (confidence: {confidence:.3f})")
    print(f"Probabilities: Real={probabilities[0]:.3f}, Fake={probabilities[1]:.3f}")
    
    # Generate Grad-CAM for predicted class
    print("Generating Grad-CAM for predicted class...")
    cams_predicted = model.generate_gradcam(
        data_dict, 
        target_class=predicted_class,
        save_visualizations=True,
        filename_prefix="predicted_class"
    )
    
    # Generate Grad-CAM for both classes
    print("Generating Grad-CAM for both classes...")
    for class_idx in [0, 1]:  # Real=0, Fake=1
        class_name = "real" if class_idx == 0 else "fake"
        print(f"  Generating for {class_name} class...")
        
        cams = model.generate_gradcam(
            data_dict,
            target_class=class_idx,
            save_visualizations=True,
            filename_prefix=f"class_{class_name}"
        )
    
    # Create comparison visualization
    print("Creating comparison visualization...")
    create_comparison_plot(
        original_image, 
        image_tensor, 
        model, 
        data_dict, 
        output_dir,
        predicted_class,
        confidence
    )
    
    print(f"All visualizations saved to: {output_dir}")


def create_comparison_plot(original_image, image_tensor, model, data_dict, 
                          output_dir, predicted_class, confidence):
    """Create a comparison plot showing original image and Grad-CAM for different layers."""
    
    # Generate CAMs for multiple layers
    layer_names = model.gradcam_layers[:3]  # Use first 3 layers
    
    fig, axes = plt.subplots(2, len(layer_names) + 1, figsize=(4 * (len(layer_names) + 1), 8))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title('Original Image')  
    axes[1, 0].axis('off')
    
    # Generate and display CAMs for each layer
    for i, layer_name in enumerate(layer_names):
        # For predicted class
        cams_pred = model.generate_gradcam(data_dict, target_class=predicted_class, layer_name=layer_name)
        if layer_name in cams_pred:
            cam = cams_pred[layer_name][0]  # First sample
            
            # Resize CAM to match original image
            input_size = original_image.shape[:2]
            cam_resized = F.interpolate(
                cam.unsqueeze(0), 
                size=input_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            
            # Create visualization
            visualization = model.gradcam.visualize_cam(original_image, cam_resized)
            
            axes[0, i + 1].imshow(visualization)
            axes[0, i + 1].set_title(f'Pred Class\n{layer_name.split(".")[-1]}')
            axes[0, i + 1].axis('off')
        
        # For opposite class
        opposite_class = 1 - predicted_class
        cams_opp = model.generate_gradcam(data_dict, target_class=opposite_class, layer_name=layer_name)
        if layer_name in cams_opp:
            cam = cams_opp[layer_name][0]  # First sample
            
            # Resize CAM
            cam_resized = F.interpolate(
                cam.unsqueeze(0), 
                size=input_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            
            # Create visualization
            visualization = model.gradcam.visualize_cam(original_image, cam_resized)
            
            axes[1, i + 1].imshow(visualization)
            axes[1, i + 1].set_title(f'Opposite Class\n{layer_name.split(".")[-1]}')
            axes[1, i + 1].axis('off')
    
    plt.suptitle(f'Grad-CAM Comparison\nPredicted: Class {predicted_class} (conf: {confidence:.3f})', 
                 fontsize=16)
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join(output_dir, 'gradcam_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Demo for CLIP Detector")
    parser.add_argument('--input', '-i', required=True, help='Path to input image')
    parser.add_argument('--output', '-o', default='./gradcam_output', help='Output directory')
    parser.add_argument('--config', '-c', help='Path to config file (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        return
    
    try:
        demonstrate_gradcam(args.input, args.output, args.config)
        print("Demo completed successfully!")
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
