import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class LayerCAM:
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

        weighted_maps = F.relu(gradients) * activations  # ReLU on gradients first (per paper)
        cam = torch.sum(weighted_maps, dim=0)
        
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
    
    def generate_layer_specific_cam(self, input_tensor, target_class_idx, layer_names):
        layer_cams = {}
        
        for layer_name in layer_names:
            temp_layercam = LayerCAM(self.model, layer_name)
            cam = temp_layercam.generate_cam(input_tensor, target_class_idx)
            layer_cams[layer_name] = cam
        max_h, max_w = 0, 0
        for cam in layer_cams.values():
            h, w = cam.shape
            max_h, max_w = max(max_h, h), max(max_w, w)
        
        combined_cam = np.zeros((max_h, max_w))
        for cam in layer_cams.values():
            cam_resized = cv2.resize(cam, (max_w, max_h))
            combined_cam += cam_resized
        
        combined_cam /= len(layer_cams)
        
        if combined_cam.max() > combined_cam.min():
            combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min())
        
        return combined_cam, layer_cams
    
    def generate_progressive_cam(self, input_tensor, target_class_idx, layers_progression):
        progressive_cams = {}
        
        for layer_name in layers_progression:
            temp_layercam = LayerCAM(self.model, layer_name)
            cam = temp_layercam.generate_cam(input_tensor, target_class_idx)
            progressive_cams[layer_name] = cam
        
        return progressive_cams

def visualize_layercam(image, cam_dict, class_names, save_path=None, alpha=0.4):
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

def compare_gradcam_layercam(image, gradcam_dict, layercam_dict, class_names, save_path=None):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    h, w = image.shape[:2]
    
    common_classes = set(gradcam_dict.keys()) & set(layercam_dict.keys())
    common_classes = sorted(list(common_classes))
    
    if not common_classes:
        print("No common classes found between GradCAM and LayerCAM")
        return
    
    num_classes = len(common_classes)
    cols = 3
    rows = num_classes
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_idx in enumerate(common_classes):
        axes[i, 0].imshow(image)
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        axes[i, 0].set_title(f'Original\n{class_name}')
        axes[i, 0].axis('off')
        
        gradcam = gradcam_dict[class_idx]
        gradcam_resized = cv2.resize(gradcam, (w, h))
        gradcam_heatmap = cv2.applyColorMap((gradcam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        gradcam_heatmap = cv2.cvtColor(gradcam_heatmap, cv2.COLOR_BGR2RGB)
        gradcam_overlay = image * 0.6 + gradcam_heatmap * 0.4
        gradcam_overlay = np.clip(gradcam_overlay, 0, 255).astype(np.uint8)
        
        axes[i, 1].imshow(gradcam_overlay)
        axes[i, 1].set_title(f'GradCAM\n(Max: {gradcam.max():.3f})')
        axes[i, 1].axis('off')
        
        layercam = layercam_dict[class_idx]
        layercam_resized = cv2.resize(layercam, (w, h))
        layercam_heatmap = cv2.applyColorMap((layercam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        layercam_heatmap = cv2.cvtColor(layercam_heatmap, cv2.COLOR_BGR2RGB)
        layercam_overlay = image * 0.6 + layercam_heatmap * 0.4
        layercam_overlay = np.clip(layercam_overlay, 0, 255).astype(np.uint8)
        
        axes[i, 2].imshow(layercam_overlay)
        axes[i, 2].set_title(f'LayerCAM\n(Max: {layercam.max():.3f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_cam_differences(gradcam, layercam):
    analysis = {}
    
    if gradcam.shape != layercam.shape:
        layercam = cv2.resize(layercam, (gradcam.shape[1], gradcam.shape[0]))

    correlation = np.corrcoef(gradcam.flatten(), layercam.flatten())[0, 1]
    analysis['correlation'] = correlation

    diff = np.abs(gradcam - layercam)
    analysis['mean_difference'] = diff.mean()
    analysis['max_difference'] = diff.max()
    analysis['std_difference'] = diff.std()

    gradcam_threshold = gradcam.mean() + gradcam.std()
    layercam_threshold = layercam.mean() + layercam.std()
    
    gradcam_focused = (gradcam > gradcam_threshold).sum()
    layercam_focused = (layercam > layercam_threshold).sum()
    
    analysis['gradcam_focused_pixels'] = gradcam_focused
    analysis['layercam_focused_pixels'] = layercam_focused
    analysis['focus_ratio'] = layercam_focused / gradcam_focused if gradcam_focused > 0 else float('inf')
    
    return analysis
