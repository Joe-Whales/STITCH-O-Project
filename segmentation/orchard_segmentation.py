import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom Dataset
class OrchardDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((224, 224))(mask)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

# Load your data
# You'll need to replace these with your actual image and mask paths
image_paths = ['path/to/image1.tif', 'path/to/image2.tif', ...]
mask_paths = ['path/to/mask1.png', 'path/to/mask2.png', ...]

dataset = OrchardDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load pretrained model
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # 2 classes: background and orchard
model = model.to(device)

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    total_iou = 0
    num_batches = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            predicted_masks = torch.argmax(outputs, dim=1)
            iou = (predicted_masks & masks).float().sum() / ((predicted_masks | masks).float().sum() + 1e-8)
            total_iou += iou.item()
            num_batches += 1
    return total_iou / num_batches

# Evaluate initial performance
initial_iou = evaluate(model, dataloader)
print(f"Initial IoU: {initial_iou:.4f}")

# Fine-tuning
num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    # Evaluate after each epoch
    iou = evaluate(model, dataloader)
    print(f"IoU after epoch {epoch+1}: {iou:.4f}")

# Final evaluation
final_iou = evaluate(model, dataloader)
print(f"Final IoU: {final_iou:.4f}")

def segment_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # Create a color mask
    color_mask = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)
    color_mask[output_predictions == 1] = [0, 255, 0]  # Green for orchards
    
    # Overlay the mask on the original image
    result = Image.fromarray(color_mask).resize(image.size)
    image = image.convert("RGBA")
    result = result.convert("RGBA")
    
    final_image = Image.alpha_composite(image, result)
    return final_image

# Segment a new image
new_image_path = 'path/to/new_image.tif'
segmented_image = segment_image(model, new_image_path)
segmented_image.save('segmented_orchard.png')