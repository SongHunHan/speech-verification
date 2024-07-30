import os
import argparse
import yaml
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from transformers import AutoProcessor, Wav2Vec2Model, Wav2Vec2ForXVector, AutoFeatureExtractor
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

from models import load_model
from dataset.VoiceDataset import load_custom_dataset
from utils.logger import CustomLogger
from utils.losses import load_losses


def get_argparse():
    parser = argparse.ArgumentParser(description='config file path')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()
    return args
    

def get_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    return config


def set_seed(seed_number):
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.deterministic = False ## 재현을 위한 거면 True
    torch.backends.cudnn.benchmark = True      ## 재현을 위한 거면 False
    

def calculate_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2).cpu().numpy()

    
def train_epoch(model, dataloader, loss_fn, optimizer, epoch, scaler, scheduler, logger, device, config):
    model.train()
    total_train_loss = 0.0
    
    train_progress_bar = tqdm(dataloader, desc=f"Training {epoch+1}", ncols=100)
    for idx, batch in enumerate(train_progress_bar):
        optimizer.zero_grad()
        
        anchor, positive, negative = batch
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config['is_amp']):
            anchor_embeddings = model(anchor).last_hidden_state.mean(dim=1)
            positive_embeddings = model(positive).last_hidden_state.mean(dim=1)
            negative_embeddings = model(negative).last_hidden_state.mean(dim=1)

        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step() ## inner batch or outer batch which is best?
        
        total_train_loss += loss.item()
        train_loss = total_train_loss / (idx + 1)
        train_progress_bar.set_postfix({"train_loss": train_loss})
        
        if idx != 0 and idx % config['log_interval'] == 0:
            logger.log({"train_loss": train_loss})
            
    train_avg_loss = total_train_loss / len(dataloader)
    return train_avg_loss


def valid_epoch(model, dataloader, loss_fn, optimizer, epoch, logger, device, config):
    model.eval()
    total_val_loss = 0
    similarities = []
    labels = []
 
    valid_progress_bar = tqdm(dataloader, desc=f"Validation {epoch+1}", ncols=100)
    for idx, batch in enumerate(valid_progress_bar):
        anchor, positive, negative = batch
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        with torch.no_grad():
            anchor_embeddings = model(anchor).last_hidden_state.mean(dim=1)
            positive_embeddings = model(positive).last_hidden_state.mean(dim=1)
            negative_embeddings = model(negative).last_hidden_state.mean(dim=1)

        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        total_val_loss += loss.item()
        valid_loss = total_val_loss / (idx + 1)
        
        pos_similirarity = calculate_similarity(anchor_embeddings, positive_embeddings)
        neg_similirarity = calculate_similarity(anchor_embeddings, negative_embeddings)
        
        similarities.extend(pos_similirarity)
        similarities.extend(neg_similirarity)
        labels.extend([1] * len(pos_similirarity))
        labels.extend([0] * len(neg_similirarity))
        
        valid_progress_bar.set_postfix({"valid_loss": valid_loss})
        
        if idx != 0 and idx % config['log_interval'] == 0:
            logger.log({'valid_loss': valid_loss})
        
    val_avg_loss = total_val_loss / len(dataloader)
    
    # Calculate ROC AUC
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Youden's J statistic to find the optimal threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_threshold = thresholds[ix]

    # Calculate F1 score at the optimal threshold
    binary_predictions = (similarities > best_threshold).astype(int)
    f1 = f1_score(labels, binary_predictions)

    logger.log({
        'valid_loss': val_avg_loss,
        'valid_roc_auc': roc_auc,
        'valid_best_threshold': best_threshold,
        'valid_f1_score': f1
    })
    print(f'Validation Loss: {val_avg_loss:.4f}, ROC AUC: {roc_auc:.4f}, Best Threshold: {best_threshold:.4f}, F1 Score: {f1:.4f}')
    return val_avg_loss, roc_auc, best_threshold, f1


def main():
    args = get_argparse()
    config_path = args.config
    config = get_config(config_path)
    
    set_seed(config['seed_number'])
    save_path = f"finetuned_model/{config['wandb']['run_name']}_{config['model_name'].split('/')[-1]}"
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config['model_name']).from_pretrained(config['model_name'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    loss_fn = load_losses(config['loss_function'])()

    train_dataset = load_custom_dataset(config['dataset_name'])(config, mode='train')
    valid_dataset = load_custom_dataset(config['dataset_name'])(config, mode='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    scaler = GradScaler(enabled=config['is_amp'])
    num_training_steps = len(train_dataloader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_training_steps/10)
    
    logger = CustomLogger(config=config)
    logger.watch(model)
    
    # best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        train_avg_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, epoch, scaler, scheduler, logger, device, config)
        val_avg_loss, roc_auc, best_threshold, f1 = valid_epoch(model, valid_dataloader, loss_fn, optimizer, epoch, logger, device, config)

        torch.save(model.state_dict(), f"{save_path}/best_model_{epoch}.pth")
        print(f"Model saved at epoch {epoch+1} with validation loss: {val_avg_loss:.4f}")

    logger.finish()
    
    
if __name__ == "__main__":
    main()
    
    