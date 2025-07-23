"""
Layer inspection utility for CLIP detector.
This script helps you identify the correct layer names for Grad-CAM visualization.

Usage:
    python inspect_clip_layers.py
"""

import torch
from transformers import CLIPModel

def inspect_clip_layers():
    """Inspect and print all layer names in the CLIP model."""
    
    print("Loading CLIP model...")
    
    # Load CLIP model (using large version as in your detector)
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name)
    vision_model = model.vision_model
    
    print(f"Model: {model_name}")
    print("="*60)
    
    print("\nVISION MODEL ARCHITECTURE:")
    print("-" * 40)
    
    layer_info = []
    
    for name, module in vision_model.named_modules():
        # Skip empty names and leaf modules without parameters
        if name and len(list(module.children())) > 0:
            layer_info.append((name, str(type(module).__name__)))
    
    # Print organized layer information
    current_prefix = ""
    for name, module_type in layer_info:
        parts = name.split(".")
        
        if len(parts) == 1:
            print(f"\n📁 {name} ({module_type})")
            current_prefix = name
        elif len(parts) == 2:
            print(f"  ├── {parts[1]} ({module_type})")
        elif len(parts) >= 3:
            if parts[0] != current_prefix:
                print(f"\n📁 {parts[0]} ({module_type})")
                current_prefix = parts[0]
            indent = "  " + "│   " * (len(parts) - 2) + "├── "
            print(f"{indent}{'.'.join(parts[1:])} ({module_type})")
    
    print("\n" + "="*60)
    print("RECOMMENDED GRAD-CAM LAYERS:")
    print("-" * 40)
    
    # Get the total number of encoder layers
    num_layers = len(vision_model.encoder.layers)
    
    recommended_layers = [
        f"encoder.layers.{num_layers-1}",  # Last layer
        f"encoder.layers.{num_layers-4}",  # 4th from last
        f"encoder.layers.{num_layers//2}",  # Middle layer
        f"encoder.layers.{num_layers//4}",  # Quarter layer
    ]
    
    print("For multi-layer visualization, use these layers:")
    for i, layer in enumerate(recommended_layers, 1):
        print(f"  {i}. {layer}")
    
    print(f"\nTotal encoder layers: {num_layers}")
    
    print("\n" + "="*60)
    print("CONFIGURATION EXAMPLE:")
    print("-" * 40)
    
    print("Add this to your config YAML file:")
    print("```yaml")
    print("gradcam_layers:")
    for layer in recommended_layers:
        print(f"  - \"{layer}\"")
    print("```")
    
    print("\n" + "="*60)
    print("LAYER SELECTION TIPS:")
    print("-" * 40)
    
    print("• Later layers (high numbers): Capture semantic, high-level features")
    print("• Earlier layers (low numbers): Capture low-level visual patterns")  
    print("• For deepfake detection, try layers from different depths")
    print("• Start with the last layer and work backwards")
    print(f"• Available range: encoder.layers.0 to encoder.layers.{num_layers-1}")


def create_layer_config_file():
    """Create a sample configuration file with layer names."""
    
    config_content = f"""# CLIP Grad-CAM Layer Configuration
# Generated automatically by inspect_clip_layers.py

gradcam_enabled: true

# Recommended layers for visualization
gradcam_layers:
  - "encoder.layers.23"    # Last transformer layer (semantic features)
  - "encoder.layers.20"    # Late layer (high-level patterns) 
  - "encoder.layers.15"    # Mid layer (intermediate features)
  - "encoder.layers.10"    # Earlier layer (low-level patterns)
  - "encoder.layers.5"     # Early layer (basic visual features)

# Alternative focused selection (fewer layers for faster processing)
# gradcam_layers:
#   - "encoder.layers.23"    # Last layer only
#   - "encoder.layers.15"    # Middle layer for comparison

gradcam_save_dir: "./gradcam_visualizations"

# Additional options
gradcam_colormap: "jet"      # Options: jet, viridis, plasma, inferno
gradcam_alpha: 0.4           # Overlay transparency (0.0 to 1.0)
"""
    
    with open("gradcam_layers_config.yaml", "w") as f:
        f.write(config_content)
    
    print(f"\nConfiguration file saved: gradcam_layers_config.yaml")


if __name__ == "__main__":
    inspect_clip_layers()
    
    create_config = input("\nCreate sample config file? (y/n): ").lower().strip()
    if create_config == 'y':
        create_layer_config_file()
    
    print("\nInspection complete!")
