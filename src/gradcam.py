import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        
        self.gradients = None
        self.activations = None
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        target_layer = self._get_layer_by_name(self.target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer {self.target_layer_name} not found")
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def _get_layer_by_name(self, layer_name):
        parts = layer_name.split('.')
        current = self.model
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, nn.Module) and part in current._modules:
                current = current._modules[part]
            else:
                return None
                
        return current
    
    def generate_cam(self, input_tensor, target_class_idx):
        output, feature_maps = self.model.get_feature_maps(input_tensor)
        self.model.zero_grad()
        
        target_output = output[0, target_class_idx]
        target_output.backward(retain_graph=True)
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        
        cam = torch.sum(weights[:, None, None] * activations, dim=0)
        cam = F.relu(cam)
        
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.detach().cpu().numpy()
    
    def generate_multi_class_cam(self, input_tensor, class_indices=None):
        if class_indices is None:
            with torch.no_grad():
                output, _ = self.model.get_feature_maps(input_tensor)
                num_classes = output.shape[1]
            class_indices = list(range(num_classes))
        
        cams = {}
        for class_idx in class_indices:
            cam = self.generate_cam(input_tensor, class_idx)
            cams[class_idx] = cam
            
        return cams

def visualize_cam(image, cam_dict, class_names, save_path=None, alpha=0.4):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    h, w = image.shape[:2]
    
    num_classes = len(cam_dict)
    cols = min(4, num_classes)
    rows = (num_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (class_idx, cam) in enumerate(cam_dict.items()):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        cam_resized = cv2.resize(cam, (w, h))
        
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = image * (1 - alpha) + heatmap * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        ax.imshow(overlay)
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        ax.set_title(f'{class_name}\n(Max: {cam.max():.3f})')
        ax.axis('off')
    for i in range(num_classes, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def evaluate_cam_quality(cam, bbox=None):
    metrics = {}
    metrics['mean'] = cam.mean()
    metrics['std'] = cam.std()
    metrics['max'] = cam.max()
    metrics['min'] = cam.min()
    
    threshold = cam.mean() + cam.std()
    focused_pixels = (cam > threshold).sum()
    total_pixels = cam.size
    metrics['concentration'] = focused_pixels / total_pixels
    metrics['sparsity'] = 1.0 - (cam > 0.01).sum() / total_pixels
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        h, w = cam.shape
        
        bbox_mask = np.zeros((h, w), dtype=bool)
        bbox_mask[y1:y2, x1:x2] = True
        
        high_activation = cam > cam.mean()
        intersection = (high_activation & bbox_mask).sum()
        union = (high_activation | bbox_mask).sum()
        metrics['bbox_iou'] = intersection / union if union > 0 else 0.0
    
    return metrics
