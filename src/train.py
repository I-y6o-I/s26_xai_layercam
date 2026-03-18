import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import os
from tqdm import tqdm
import argparse

from model import CheXpertResNet50
from dataset import CheXpertDataset
from loss import MaskedBCELoss, MaskedFocalLoss
from preprocess import preprocess_chexpert_dataframe

class CheXpertTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CheXpertResNet50(num_classes=config['num_classes']).to(self.device)
        
        if config['loss_type'] == 'bce':
            self.criterion = MaskedBCELoss()
        elif config['loss_type'] == 'focal':
            self.criterion = MaskedFocalLoss(alpha=config['alpha'], gamma=config['gamma'])
        else:
            raise ValueError(f"Unknown loss type: {config['loss_type']}")
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        self.best_val_auc = 0.0
        self.best_val_f1 = 0.0
        
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        all_masks = []
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, targets, valid_mask) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            valid_mask = valid_mask.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            
            loss = self.criterion(logits, targets, valid_mask)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            with torch.no_grad():
                predictions = torch.sigmoid(logits)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_masks.append(valid_mask.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        all_masks = np.concatenate(all_masks)
        
        metrics = self.calculate_metrics(all_predictions, all_targets, all_masks)
        
        return epoch_loss / len(dataloader), metrics
    
    def validate_epoch(self, dataloader):
        self.model.eval()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for images, targets, valid_mask in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                valid_mask = valid_mask.to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, targets, valid_mask)
                
                epoch_loss += loss.item()
                
                predictions = torch.sigmoid(logits)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_masks.append(valid_mask.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        all_masks = np.concatenate(all_masks)
        
        metrics = self.calculate_metrics(all_predictions, all_targets, all_masks)
        
        return epoch_loss / len(dataloader), metrics
    
    def calculate_metrics(self, predictions, targets, masks):
        metrics = {}
        
        class_aucs = []
        class_f1s = []
        
        for i in range(targets.shape[1]):
            valid_indices = masks[:, i]
            if valid_indices.sum() > 1:
                class_preds = predictions[valid_indices, i]
                class_targets = targets[valid_indices, i]
                
                if len(np.unique(class_targets)) > 1:
                    auc = roc_auc_score(class_targets, class_preds)
                    class_aucs.append(auc)
                
                class_pred_labels = (class_preds > 0.5).astype(int)
                f1 = f1_score(class_targets, class_pred_labels, zero_division=0)
                class_f1s.append(f1)
        
        metrics['auc_mean'] = np.mean(class_aucs) if class_aucs else 0.0
        metrics['f1_mean'] = np.mean(class_f1s) if class_f1s else 0.0
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            self.scheduler.step(val_loss)
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('AUC/Train', train_metrics['auc_mean'], epoch)
            self.writer.add_scalar('AUC/Val', val_metrics['auc_mean'], epoch)
            self.writer.add_scalar('F1/Train', train_metrics['f1_mean'], epoch)
            self.writer.add_scalar('F1/Val', val_metrics['f1_mean'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train AUC: {train_metrics['auc_mean']:.4f}, Val AUC: {val_metrics['auc_mean']:.4f}")
            print(f"Train F1: {train_metrics['f1_mean']:.4f}, Val F1: {val_metrics['f1_mean']:.4f}")
            
            if val_metrics['auc_mean'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc_mean']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_auc': self.best_val_auc,
                    'config': self.config
                }, os.path.join(self.config['checkpoint_dir'], 'best_model.pth'))
                print(f"New best model saved with AUC: {self.best_val_auc:.4f}")
        
        self.writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to CheXpert data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'])
    parser.add_argument('--alpha', type=float, default=1.0, help='Focal loss alpha')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    target_cols = [
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
        'Support Devices'
    ]
    
    train_df = preprocess_chexpert_dataframe(
        args.data_root, 'train.csv', target_cols
    )
    val_df = preprocess_chexpert_dataframe(
        args.data_root, 'valid.csv', target_cols
    )
    
    train_dataset = CheXpertDataset(train_df, target_cols)
    val_dataset = CheXpertDataset(val_df, target_cols)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    config = {
        'num_classes': len(target_cols),
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'loss_type': args.loss_type,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'log_dir': args.log_dir,
        'checkpoint_dir': args.checkpoint_dir
    }
    
    trainer = CheXpertTrainer(config)
    trainer.train(train_loader, val_loader, args.num_epochs)

if __name__ == '__main__':
    main()
