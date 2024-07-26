import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import os
import logging
import yaml
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import numpy as np
import argparse 
from custom_dataset import build_custom_dataloader
from utils import performances


# Import the custom dataset and dataloader
from custom_dataset import build_custom_dataloader

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        efficientnet = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        
        # Only use the first four layers
        self.features = nn.Sequential(
            efficientnet.features[0],
            efficientnet.features[1],
            efficientnet.features[2],
            efficientnet.features[3]
        )
        
        # Freeze the parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.features(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch: int = 56, out_ch: int = 56):  # EfficientNet-B4 4th layer output features
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3 = DoubleConv(768, 256)
        self.dec2 = DoubleConv(384, 128)
        self.dec1 = DoubleConv(128 + 64 + in_ch, 64)

        self.out_conv = nn.Conv2d(64, out_ch, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, output_padding=1)  # Custom upsampling
        self.upsample2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # Custom upsampling
        self.upsample1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # Custom upsampling

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d3 = self.dec3(torch.cat([self.upsample3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample1(d2), e1, x], dim=1))

        return self.out_conv(d1)

class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.unet = UNet()

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        reconstructed = self.unet(features)
        return reconstructed, features

def train_model(model: nn.Module, train_loader, num_epochs: int, learning_rate: float, weight_decay: float, device: str, model_path: str, checkpoint = None):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("Model loaded from checkpoint.")

    best_train_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training:")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        for i, input in progress_bar:
            images = input["image"].to(device)
            
            optimizer.zero_grad()
            with autocast():
                reconstructed_features, features = model(images)
                loss = criterion(reconstructed_features, features.detach())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            progress_bar.set_description(f"Train Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader.dataset)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, model_path)
            print(f'Model saved at epoch {epoch+1}, with train loss: {train_loss:.4f}')

def validate(val_loader, model, verbose=False):
    model.eval()
    criterion = nn.MSELoss()
    classes = {}

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            clsname = input["clsname"][0]
            if clsname not in classes.keys():
                classes[clsname] = {"losses": [], "labels": []}

            image = input["image"].cuda()
            feature_rec, features = model(image)

            classes[clsname]["losses"].append(criterion(feature_rec, features.detach()).item())
            classes[clsname]["labels"].append(input["label"].item())

    results = {}
    for clsname in classes.keys():
        print(f"\nClass: {clsname}")
        auroc, accuracy = performances(classes[clsname]["labels"], classes[clsname]["losses"], verbose=verbose)
        results[clsname] = {"auroc": auroc, "accuracy": accuracy}
    return results

def create_and_run_model(cfg, device, logger = False, eval_mode = False):    
    checkpoint = None
    model = FullModel().to(device)

    if os.path.exists(cfg["model_path"]):
        checkpoint = torch.load(cfg["model_path"])
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from", cfg["model_path"])
    else:
        print(f'No saved model found at {cfg["model_path"]}. Training new model...')

    if not eval_mode:
        train_loader = build_custom_dataloader(cfg["train"], training=True)
        train_model(model, train_loader, cfg["num_epochs"], cfg["learning_rate"], cfg["weight_decay"], device, cfg["model_path"], checkpoint=checkpoint)
    
    test_loader = build_custom_dataloader(cfg["test"], training=False)
    run_stats = validate(test_loader, model, verbose=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    create_and_run_model(cfg, device, False, True)