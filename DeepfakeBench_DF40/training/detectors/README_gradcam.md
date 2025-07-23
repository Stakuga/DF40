# Grad-CAM Integration for CLIP Deepfake Detector

This directory contains the implementation of Gradient-based Class Activation Mapping (Grad-CAM) for the CLIP-based deepfake detector in the DF40 benchmark.

## 🚀 Features

- **Grad-CAM Visualization**: Generate heatmaps showing which regions the model focuses on during prediction
- **Multi-layer Support**: Visualize attention across different transformer layers  
- **Batch Processing**: Generate visualizations for multiple samples efficiently
- **Automated Testing**: Integrate Grad-CAM into your evaluation pipeline
- **Comprehensive Output**: Save visualizations with metadata and comparison plots

## 📁 Files Overview

### Core Implementation
- `utils/gradcam.py` - Main Grad-CAM implementation with support for both Grad-CAM and Grad-CAM++
- `clip_detector_gradcam.py` - Enhanced CLIP detector with built-in Grad-CAM functionality  
- `config/detector/clip_gradcam.yaml` - Configuration file for the enhanced detector

### Demo and Testing Scripts
- `utils/gradcam_demo.py` - Standalone demo script for single image visualization
- `test_with_gradcam.py` - Enhanced testing script with Grad-CAM integration
- `README_gradcam.md` - This documentation file

## 🛠️ Setup

### 1. Dependencies
The Grad-CAM functionality requires the following packages (in addition to existing dependencies):
```bash
pip install matplotlib seaborn opencv-python pillow
```

### 2. Model Registration
Make sure the new detector is registered in your `detectors/__init__.py`:
```python
from .clip_detector_gradcam import CLIPDetectorWithGradCAM
```

## 🎯 Quick Start

### Option 1: Standalone Demo
Test Grad-CAM on a single image:

```bash
cd DeepfakeBench_DF40/training
python detectors/utils/gradcam_demo.py --input path/to/your/image.jpg --output ./gradcam_results
```

### Option 2: Integrated Testing
Run evaluation with Grad-CAM visualization:

```bash
python test_with_gradcam.py \
    --detector_path config/detector/clip_gradcam.yaml \
    --weights_path path/to/your/weights.pth \
    --test_dataset FaceForensics++ \
    --gradcam_enabled \
    --gradcam_samples 50 \
    --gradcam_output_dir ./test_gradcam_results
```

### Option 3: Programmatic Usage
Use Grad-CAM in your own code:

```python
import torch
from detectors.clip_detector_gradcam import CLIPDetectorWithGradCAM

# Initialize model with Grad-CAM enabled
config = {
    'loss_func': 'cross_entropy_loss',
    'gradcam_enabled': True,
    'gradcam_layers': ['encoder.layers.23', 'encoder.layers.20'],
    'gradcam_save_dir': './my_gradcam_output'
}

model = CLIPDetectorWithGradCAM(config)
model.eval()

# Load your image (assuming preprocessed)
data_dict = {'image': your_image_tensor}

# Generate Grad-CAM visualizations
cams = model.generate_gradcam(
    data_dict, 
    save_visualizations=True,
    filename_prefix="my_analysis"
)

# The visualizations are automatically saved to the configured directory
```

## ⚙️ Configuration Options

### Grad-CAM Settings in YAML Config
```yaml
# Enable/disable Grad-CAM
gradcam_enabled: true

# Target layers for visualization
gradcam_layers:
  - "encoder.layers.23"    # Last transformer layer
  - "encoder.layers.20"    # Earlier layer for comparison  
  - "encoder.layers.15"    # Even earlier layer
  - "encoder.layers.10"    # Mid-level features

# Output directory for visualizations
gradcam_save_dir: "./gradcam_visualizations"
```

### Command Line Options
```bash
--gradcam_enabled                    # Enable Grad-CAM during testing
--gradcam_samples 50                # Number of samples to visualize  
--gradcam_output_dir ./results      # Output directory
--save_all_predictions              # Save visualizations for all samples (not just wrong predictions)
```

## 🎨 Output Examples

The Grad-CAM visualization generates several types of outputs:

### 1. Individual Layer Visualizations
- `gradcam_encoder.layers.23_sample0.png` - Heatmap overlay for last transformer layer
- `gradcam_encoder.layers.20_sample0.png` - Heatmap overlay for earlier layer
- Each shows which regions influenced the model's decision

### 2. Comparison Plots
- `gradcam_comparison.png` - Side-by-side comparison of multiple layers
- Shows how attention patterns differ across network depth

### 3. Metadata Files
- `sample0_metadata.txt` - Contains prediction details:
  ```
  True Label: 1 (Fake)
  Predicted Label: 0 (Real)  
  Confidence: 0.7234
  Correct: False
  Dataset: FaceForensics++
  ```

## 🔍 Understanding the Visualizations

### Heatmap Colors
- **Red/Yellow**: High activation regions (model focuses here)
- **Blue/Purple**: Low activation regions (model ignores these)
- **Green**: Medium activation regions

### Layer Interpretation
- **Later layers** (e.g., encoder.layers.23): Capture high-level semantic features
- **Earlier layers** (e.g., encoder.layers.10): Capture low-level visual patterns
- **Compare layers** to understand the model's decision-making process

### Typical Patterns
- **Real faces**: Often focus on natural facial features, skin texture
- **Fake faces**: May focus on artifacts, unnatural transitions, compression artifacts
- **Misclassifications**: Look for unexpected attention patterns

## 🚨 Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed and paths are correct
2. **CUDA Memory**: Reduce batch size if running out of GPU memory during visualization  
3. **Layer Names**: Verify layer names match your model architecture using `model.named_modules()`
4. **No Visualizations**: Check that `gradcam_enabled=True` in config

### Debug Layer Names
```python
# Print all available layer names
for name, module in model.named_modules():
    print(name)
```

### Memory Management
```python
# For large batches, process samples individually
for i in range(batch_size):
    single_data = {'image': batch_data['image'][i:i+1]}
    cams = model.generate_gradcam(single_data)
```

## 📊 Performance Considerations

- **Memory Usage**: Grad-CAM requires storing gradients, increasing memory usage by ~30%
- **Speed**: Visualization adds overhead; disable during training if not needed
- **Storage**: Each visualization can be 1-5MB; plan storage accordingly

## 🤝 Integration with Existing Workflow

### Training Integration
To use during training, modify your training loop:

```python
# During validation
if epoch % 5 == 0:  # Every 5 epochs
    model.enable_gradcam()
    # Run some validation samples with visualization
    model.disable_gradcam()  # Disable to save memory
```

### Evaluation Integration
The `test_with_gradcam.py` script is designed to be a drop-in replacement for your existing test script with added visualization capabilities.

## 📚 References

1. **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
2. **Grad-CAM++**: Chattopadhay et al. "Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks" (WACV 2018)
3. **CLIP**: Radford et al. "Learning Transferable Visual Representations" (ICML 2021)

## 📝 Citation

If you use this Grad-CAM implementation in your research, please cite the DF40 paper and the original Grad-CAM paper:

```bibtex
@inproceedings{yan2024df40,
  title={DF40: Toward Next-Generation Deepfake Detection},
  author={Yan, Zhiyuan and others},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2024}
}

@inproceedings{selvaraju2017grad,
  title={Grad-CAM: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasaath R and others},
  booktitle={ICCV},
  year={2017}
}
```
