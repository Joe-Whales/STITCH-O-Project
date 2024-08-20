import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the SAM model using the downloaded checkpoint
sam_checkpoint = "sam_vit_l_0b3195.pth"  # Update this path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = "vit_l"  # Model type corresponding to the checkpoint
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Initialize the SAM automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam, pred_iou_thresh=0.12, stability_score_thresh=0.88, box_nms_thresh=0.4, crop_nms_thresh=0.4)

# Load and preprocess the input image
image_path = "1.png"  # Replace with the path to your image
image = np.array(Image.open(image_path).convert("RGB"))

# Generate masks for the image
masks = mask_generator.generate(image)

def show_masks(image, masks, random_colors=True):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    # Create a color map
    if random_colors:
        color_map = plt.cm.get_cmap('tab20')  # You can try other colormaps like 'Set1', 'Set2', 'Set3', etc.
    else:
        color_map = plt.cm.get_cmap('viridis')  # A sequential colormap
    
    # Create a single mask that combines all individual masks
    combined_mask = np.zeros(image.shape[:2] + (4,), dtype=np.float32)
    
    for i, mask in enumerate(masks):
        mask_image = mask['segmentation']
        if random_colors:
            color = color_map(random.random())
        else:
            color = color_map(i / len(masks))
        
        mask_color = np.concatenate([color[:3], [1.0]])  # RGBA
        combined_mask[mask_image] = mask_color
    
    plt.imshow(combined_mask)
    plt.title(f"Number of segments: {len(masks)}")
    plt.axis('off')
    plt.show()

# Generate masks for the image
masks = mask_generator.generate(image)

# Show masks with random colors
show_masks(image, masks, random_colors=True)