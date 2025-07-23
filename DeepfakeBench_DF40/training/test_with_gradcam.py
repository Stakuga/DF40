"""
Test script with Grad-CAM visualization support for CLIP detector.
Modified version of the original test.py to include Grad-CAM functionality.

Usage:
    python test_with_gradcam.py --detector_path config/detector/clip_gradcam.yaml --weights_path path/to/weights.pth --test_dataset FaceForensics++ --gradcam_samples 10
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

# Add Grad-CAM specific imports
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='Test model with Grad-CAM visualization.')
parser.add_argument('--detector_path', type=str, 
                    default='./config/detector/clip_gradcam.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+", default=["FaceForensics++"])
parser.add_argument('--weights_path', type=str, 
                    help='Path to model weights')
parser.add_argument('--gradcam_enabled', action='store_true', default=False,
                    help='Enable Grad-CAM visualization during testing')
parser.add_argument('--gradcam_samples', type=int, default=10,
                    help='Number of samples to generate Grad-CAM for')
parser.add_argument('--gradcam_output_dir', type=str, default='./test_gradcam_output',
                    help='Directory to save Grad-CAM visualizations')
parser.add_argument('--save_all_predictions', action='store_true', default=False,
                    help='Save Grad-CAM for all predictions (not just wrong ones)')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['manualSeed'])

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # build up data set  -->   train_data_loader
        config_copy = deepcopy(config)
        config_copy['dataset']['test_batchSize'] = max(config['dataset']['test_batchSize'] // 2, 1)
        test_set = DeepfakeAbstractBaseDataset(
                config=config_copy['dataset'],
                mode='test', 
        )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config_copy['dataset']['test_batchSize'],
                shuffle=False, 
                num_workers=int(config_copy['dataset']['workers']),
                collate_fn=test_set.collate_fn,
            )
        return test_data_loader
    
    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        config['dataset']['test_dataset'] = one_test_name
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name) 
    return test_data_loaders

def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['auc', 'acc', 'eer', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring

def test_epoch_with_gradcam(model, test_data_loaders, config, logger, gradcam_config):
    """Test model with Grad-CAM visualization support."""
    
    # Extract Grad-CAM configuration
    gradcam_enabled = gradcam_config['enabled']
    gradcam_samples = gradcam_config['samples']
    gradcam_output_dir = gradcam_config['output_dir']
    save_all_predictions = gradcam_config['save_all']
    
    # Create Grad-CAM output directory
    if gradcam_enabled:
        os.makedirs(gradcam_output_dir, exist_ok=True)
        logger.info(f"Grad-CAM visualizations will be saved to: {gradcam_output_dir}")
    
    model.eval()
    metric_scoring = choose_metric(config)
    
    # Testing results storage
    results_dict = defaultdict(dict)
    gradcam_samples_processed = 0
    
    for test_name, test_data_loader in test_data_loaders.items():
        logger.info(f"Testing on {test_name} dataset...")
        
        # Storage for this dataset
        predictions_list = []
        labels_list = []
        prob_list = []
        
        # Progress bar
        pbar = tqdm(test_data_loader, desc=f'Testing {test_name}')
        
        for batch_idx, data_dict in enumerate(pbar):
            # Move to device
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(device)
            
            batch_size = data_dict['image'].shape[0]
            labels = data_dict['label']
            
            with torch.no_grad():
                predictions = model(data_dict, inference=True)
                prob = predictions['prob']
                cls = predictions['cls']
            
            # Store results
            pred_classes = torch.argmax(cls, dim=1)
            predictions_list.extend(pred_classes.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            prob_list.extend(prob.cpu().numpy())
            
            # Generate Grad-CAM visualizations
            if gradcam_enabled and gradcam_samples_processed < gradcam_samples:
                generate_gradcam_for_batch(
                    model, data_dict, labels, pred_classes, prob,
                    batch_idx, test_name, gradcam_output_dir, 
                    save_all_predictions, logger
                )
                gradcam_samples_processed += batch_size
            
            # Update progress bar
            current_acc = np.mean(np.array(predictions_list) == np.array(labels_list))
            pbar.set_postfix({'Accuracy': f'{current_acc:.4f}'})
        
        # Calculate metrics for this dataset
        predictions_array = np.array(predictions_list)
        labels_array = np.array(labels_list)
        prob_array = np.array(prob_list)
        
        test_metrics = get_test_metrics(
            y_pred=predictions_array,
            y_true=labels_array,
            prob=prob_array
        )
        
        results_dict[test_name] = test_metrics
        
        # Log results
        logger.info(f"Results for {test_name}:")
        for metric_name, metric_value in test_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return results_dict

def generate_gradcam_for_batch(model, data_dict, labels, pred_classes, prob, 
                              batch_idx, test_name, output_dir, save_all, logger):
    """Generate Grad-CAM visualizations for a batch of samples."""
    
    batch_size = data_dict['image'].shape[0]
    
    for sample_idx in range(batch_size):
        # Extract single sample
        sample_data = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                sample_data[key] = value[sample_idx:sample_idx+1]
            else:
                sample_data[key] = [value[sample_idx]] if isinstance(value, list) else value
        
        true_label = labels[sample_idx].item()
        pred_label = pred_classes[sample_idx].item()
        confidence = prob[sample_idx].item()
        
        # Decide whether to save this sample
        is_correct = (true_label == pred_label)
        should_save = save_all or not is_correct
        
        if should_save and hasattr(model, 'gradcam') and model.gradcam:
            try:
                # Create filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                status = "correct" if is_correct else "wrong"
                filename_prefix = f"{test_name}_batch{batch_idx}_sample{sample_idx}_{status}_{timestamp}"
                
                # Generate Grad-CAM
                sample_output_dir = os.path.join(output_dir, test_name)
                os.makedirs(sample_output_dir, exist_ok=True)
                
                # Save basic info
                info = {
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'timestamp': timestamp
                }
                
                # Save Grad-CAM visualizations
                visualizations = model.save_gradcam_batch(
                    sample_data,
                    save_dir=sample_output_dir,
                    filename_prefix=filename_prefix
                )
                
                # Save metadata
                metadata_path = os.path.join(sample_output_dir, f"{filename_prefix}_metadata.txt")
                with open(metadata_path, 'w') as f:
                    f.write(f"True Label: {true_label} ({'Real' if true_label == 0 else 'Fake'})\n")
                    f.write(f"Predicted Label: {pred_label} ({'Real' if pred_label == 0 else 'Fake'})\n")
                    f.write(f"Confidence: {confidence:.4f}\n")
                    f.write(f"Correct: {is_correct}\n")
                    f.write(f"Dataset: {test_name}\n")
                    f.write(f"Batch: {batch_idx}, Sample: {sample_idx}\n")
                
                if not is_correct:
                    logger.info(f"Saved Grad-CAM for misclassified sample: {filename_prefix}")
                
            except Exception as e:
                logger.warning(f"Failed to generate Grad-CAM for sample {sample_idx}: {str(e)}")

def create_summary_report(results_dict, gradcam_output_dir, logger):
    """Create a summary report of test results and Grad-CAM analysis."""
    
    summary_path = os.path.join(gradcam_output_dir, "test_summary_report.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=== DEEPFAKE DETECTION TEST RESULTS WITH GRAD-CAM ===\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall results
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        
        for dataset_name, metrics in results_dict.items():
            f.write(f"\nDataset: {dataset_name}\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"  {metric_name}: {metric_value:.4f}\n")
        
        # Grad-CAM analysis info
        f.write(f"\n\nGRAD-CAM VISUALIZATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Visualizations saved to: {gradcam_output_dir}\n")
        f.write("Check subdirectories for dataset-specific visualizations.\n")
        f.write("Each visualization includes:\n")
        f.write("  - Original image with Grad-CAM overlay\n")
        f.write("  - Multiple layer visualizations\n") 
        f.write("  - Prediction metadata\n")
    
    logger.info(f"Summary report saved to: {summary_path}")

def main():
    # Load configuration
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set test datasets
    config['test_dataset'] = args.test_dataset
    
    # Enable Grad-CAM in config if requested
    if args.gradcam_enabled:
        config['gradcam_enabled'] = True
        if 'gradcam_save_dir' not in config:
            config['gradcam_save_dir'] = args.gradcam_output_dir
    
    # Initialize logger
    logger = create_logger(output_dir=args.gradcam_output_dir, name="test_gradcam")
    logger.info("Starting deepfake detection test with Grad-CAM...")
    logger.info(f"Configuration loaded from: {args.detector_path}")
    
    # Initialize random seed
    init_seed(config)
    
    # Build model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    
    # Load weights if provided
    if args.weights_path and os.path.exists(args.weights_path):
        logger.info(f"Loading weights from: {args.weights_path}")
        checkpoint = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    else:
        logger.warning("No weights loaded - using randomly initialized model")
    
    # Prepare test data
    logger.info("Preparing test data...")
    test_data_loaders = prepare_testing_data(config)
    
    # Grad-CAM configuration
    gradcam_config = {
        'enabled': args.gradcam_enabled,
        'samples': args.gradcam_samples,
        'output_dir': args.gradcam_output_dir,
        'save_all': args.save_all_predictions
    }
    
    # Run testing with Grad-CAM
    logger.info("Starting testing...")
    results = test_epoch_with_gradcam(
        model, test_data_loaders, config, logger, gradcam_config
    )
    
    # Create summary report
    if args.gradcam_enabled:
        create_summary_report(results, args.gradcam_output_dir, logger)
    
    # Print final results
    logger.info("\n" + "="*50)
    logger.info("FINAL RESULTS:")
    logger.info("="*50)
    
    for dataset_name, metrics in results.items():
        logger.info(f"\n{dataset_name}:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    logger.info("\nTesting completed!")

if __name__ == "__main__":
    main()
