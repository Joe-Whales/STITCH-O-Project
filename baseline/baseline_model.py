import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import logging
import yaml
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights, resnet50, ResNet50_Weights
import argparse 
from custom_dataset import build_custom_dataloader
from utils import performances
from custom_dataset import build_custom_dataloader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

class FeatureExtractor(nn.Module):
    """
    A feature extraction module that uses pre-trained models to extract features from input images.

    Args:
        model_name (str): The name of the pre-trained model to use. Default is 'efficientnet_b4'.

    Attributes:
        model_name (str): The name of the pre-trained model being used.
        features (nn.Sequential): The feature extraction layers of the chosen model.
        out_channels (int): The number of output channels from the feature extractor.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    def __init__(self, model_name='efficientnet_b4'):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name
        
        if model_name == 'efficientnet_b4':
            efficientnet = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            self.features = nn.Sequential(
                efficientnet.features[0],
                efficientnet.features[1],
                efficientnet.features[2],
                efficientnet.features[3]
            )
            self.out_channels = 56 
        elif model_name == 'resnet50':
            print("Restnet not supported yet")
            exit()
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1
            )
            self.out_channels = 256
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.features(x)

class DoubleConv(nn.Module):
    """
    A double convolutional block used in the U-Net architecture.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.

    Attributes:
        conv (nn.Sequential): A sequence of two convolutional layers with batch normalization and ReLU activation.
    """
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
    """
    A U-Net architecture for image segmentation or feature reconstruction.

    Args:
        in_ch (int): Number of input channels. Default is 56.
        out_ch (int): Number of output channels. Default is 56.

    Attributes:
        enc1, enc2, enc3, enc4 (DoubleConv): Encoder blocks.
        dec3, dec2, dec1 (DoubleConv): Decoder blocks.
        out_conv (nn.Conv2d): Output convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        upsample3, upsample2, upsample1 (nn.ConvTranspose2d): Upsampling layers.
    """
    def __init__(self, in_ch: int = 56, out_ch: int = 56):
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

class BaselineModel(nn.Module):
    """
    A baseline model combining a feature extractor and a U-Net for feature reconstruction.

    Args:
        feature_extractor (str): The name of the feature extractor to use. Default is 'efficientnet_b4'.

    Attributes:
        feature_extractor (FeatureExtractor): The feature extraction module.
        unet (UNet): The U-Net module for feature reconstruction.
    """
    def __init__(self, feature_extractor='efficientnet_b4'):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor(feature_extractor)
        self.unet = UNet(in_ch=self.feature_extractor.out_channels, out_ch=self.feature_extractor.out_channels)

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        reconstructed = self.unet(features)
        return reconstructed, features

def get_scheduler(optimizer, cfg):
    """
    Create a learning rate scheduler based on the configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to use with the scheduler.
        cfg (dict): A configuration dictionary containing scheduler settings.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: The created scheduler, or None if 'none' is specified.

    Raises:
        ValueError: If an unsupported scheduler name is provided.
    """
    scheduler_cfg = cfg['scheduler']
    if scheduler_cfg['name'] == 'step':
        scheduler = StepLR(optimizer, step_size=scheduler_cfg['step']['step_size'], gamma=scheduler_cfg['step']['gamma'])
    elif scheduler_cfg['name'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_cfg['plateau']['mode'], factor=scheduler_cfg['plateau']['factor'], patience=scheduler_cfg['plateau']['patience'])
    elif scheduler_cfg['name'] == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_cfg['name']}")
    return scheduler

def train_model(model: nn.Module, train_loader, val_loader, cfg, device: str, checkpoint = None):
    """
    Train the model using the provided data loaders and configuration.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        cfg (dict): Configuration dictionary containing training parameters.
        device (str): The device to use for training ('cuda' or 'cpu').
        checkpoint (dict, optional): A checkpoint to resume training from.

    Returns:
        tuple: A tuple containing the best validation accuracy and best thresholds.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scaler = GradScaler()
    scheduler = get_scheduler(optimizer, cfg)

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Model loaded from checkpoint.")

    best_accuracy = 0.0
    best_thresholds = None

    for epoch in range(cfg['num_epochs']):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{cfg['num_epochs']}")
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
        
        print(f'Epoch [{epoch+1}/{cfg['num_epochs']}], Train Loss: {train_loss:.4f}')
        logging.info(f'Epoch {epoch+1}/{cfg['num_epochs']}, Train Loss: {train_loss:.4f}')

        if (epoch+1) % 3 == 0:
            # Validate and get run stats
            run_stats = validate(val_loader, model, verbose=False)
            
            # Calculate overall accuracy
            total_correct = sum(stats['accuracy'] for stats in run_stats.values())
            overall_accuracy = total_correct / len(run_stats)
            
            # log auroc for each class
            for clsname, stats in run_stats.items():
                logging.info(f'Epoch {epoch+1}/{cfg['num_epochs']}, Class: {clsname}, AUROC: {stats["auroc"]:.4f}, Accuracy: {stats["accuracy"]:.4f}, Threshold: {stats["threshold"]:.4f}')
            
            print(f'Validation Accuracy: {overall_accuracy:.4f}')
            logging.info(f'Epoch {epoch+1}/{cfg['num_epochs']}, Validation Accuracy: {overall_accuracy:.4f}')

            if overall_accuracy > best_accuracy:
                best_accuracy = overall_accuracy
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'accuracy': best_accuracy
                }, cfg['model_path'])
                print(f'Model saved at epoch {epoch+1}, with validation accuracy: {best_accuracy:.4f}')
                
                # Save thresholds
                best_thresholds = {clsname: stats['threshold'] for clsname, stats in run_stats.items()}
                torch.save(best_thresholds, 'baseline/best_inference_thresholds.pth')
                print(f'Best thresholds saved at best_inference_thresholds.pth')
              
            if cfg['scheduler']['name'] == 'plateau':
                scheduler.step(overall_accuracy)
            elif cfg['scheduler']['name'] == 'step':
                scheduler.step()
            elif cfg['scheduler']['name'] == 'none':
                drop_count = 0
                if 0.9 > overall_accuracy > 0.85 and drop_count < 2:
                    # set optimizer learning rate to 0.5 of the original learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.7 * param_group['lr']
                    drop_count += 1
                elif overall_accuracy > 0.9 and drop_count < 4:
                    # set optimizer learning rate to 0.1 of the original learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.5 * param_group['lr']
                    drop_count += 1

    return best_accuracy, best_thresholds

def validate(val_loader, model, verbose=False):
    """
    Validate the model using the provided validation data loader.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The model to validate.
        verbose (bool): Whether to print detailed performance metrics. Default is False.

    Returns:
        dict: A dictionary containing validation results for each class.
    """
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
        auroc, accuracy, threshold = performances(classes[clsname]["labels"], classes[clsname]["losses"], verbose=verbose)
        results[clsname] = {"auroc": auroc, "accuracy": accuracy, "threshold": threshold}
    return results

def create_and_run_model(cfg, device, eval_mode = False):
    """
    Create a model, load a checkpoint if available, and either train or evaluate the model.

    Args:
        cfg (dict): Configuration dictionary containing model and data parameters.
        device (str): The device to use for computations ('cuda' or 'cpu').
        eval_mode (bool): Whether to run in evaluation mode. Default is False.

    Returns:
        dict: A dictionary containing the best thresholds for each class.
    """
    checkpoint = None
    model = BaselineModel(feature_extractor=cfg["feature_extractor"]).to(device)

    if os.path.exists(cfg["model_path"]):
        checkpoint = torch.load(cfg["model_path"])
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from", cfg["model_path"])
    else:
        print(f'No saved model found at {cfg["model_path"]}. Training new model...')

    val_loader = build_custom_dataloader(cfg["test"], training=False)
    if not eval_mode:
        train_loader = build_custom_dataloader(cfg["train"], training=True)
        best_accuracy, best_thresholds = train_model(model, train_loader, val_loader, cfg, device, checkpoint=checkpoint)
        print(f'Training completed. Best accuracy: {best_accuracy:.4f}')
        logging.info(f'Training completed. Best accuracy: {best_accuracy:.4f}')
    else:
        run_stats = validate(val_loader, model, verbose=False)
        best_thresholds = {clsname: stats['threshold'] for clsname, stats in run_stats.items()}
    
    return best_thresholds
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    logging.basicConfig(filename='baseline/training_log.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    stats = create_and_run_model(cfg, device, False, False)
    logging.info(f'Run stats: {stats}')