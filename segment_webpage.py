import os
import gc
import warnings
import random
import string
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# we use cpu then shut down sam2 warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sam2")
device = torch.device("cpu")
print(f"Using device: {device}")

# load SAM2 model
print("Loading SAM2...")
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# Configure automatic mask generator with memory-optimized settings
# reduce points_per_side and points_per_batch to save memory
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.95,
    stability_score_offset=1.0,
    crop_n_layers=0, 
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500,  # increased to filter smaller segments
    use_m2m=True,  # disabled for memory
)

# Load webpage screenshot into numpy array
image_path = 'screenshot.png'
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)
print(f"Processing image of size: {image_np.shape}")

# Generate masks
# WARNING takes forever on cpu
print("Generating masks...")
try:
    masks = mask_generator.generate(image_np)
    print(f"{len(masks)} masks.")
    
    # garbage collection
    gc.collect()
except Exception as e:
    print(f"Error: {e}")
    raise

# Initialize OpenAI client for labeling (commented out for now)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# HELPER FUNCTIONS

def get_bbox_from_mask(mask):
    """Get bounding box [x, y, w, h] from binary mask"""

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]  # [x, y, w, h]

def extract_segment_image(image_np, mask, bbox):
    """Extract the segment image with transparent background"""

    x, y, w, h = bbox
    
    # Create new RGBA image
    segment = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Crop the original image and mask to bbox
    cropped_image = image_np[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    # transpose mask (transpose different color channels)
    segment[cropped_mask, :3] = cropped_image[cropped_mask]  # RGB
    segment[cropped_mask, 3] = 255  # Alpha (opaque)
    segment[~cropped_mask, 3] = 0  # Alpha (transparent)
    
    return Image.fromarray(segment, 'RGBA')

# saving to random folder
random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
output_folder = f"segments_{random_suffix}"
os.makedirs(output_folder, exist_ok=True)
print(f"Created output folder: {output_folder}")

segments = []
max_segments = len(masks)  # Process all segments
print(f"Processing {max_segments} segments...")

# saving unit segments
for i, ann in enumerate(masks):
    if i % 10 == 0:
        print(f"Processing segment {i+1}/{max_segments}")
    
    mask = ann['segmentation']
    bbox = get_bbox_from_mask(mask)
    
    # enerate random number for filename
    random_num = random.randint(10000, 99999)
    
    # Extract and save the segment image
    segment_image = extract_segment_image(image_np, mask, bbox)
    segment_filename = f"segment_{random_num}.png"
    segment_path = os.path.join(output_folder, segment_filename)
    segment_image.save(segment_path)
    
    segments.append({
        "id": i,
        "bbox": bbox,  # see bbox func -> [x, y, w, h]
        "area": int(ann['area']),
        "filename": segment_filename
    })
    
    # garbage collection
    if i % 10 == 0:
        gc.collect()

print(f"Saved {len(segments)} segment images to {output_folder}/")

# Visualize and save overlay
def show_anns(anns, labels, image_np):
    """annotations on the image"""

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    img = np.ones((image_np.shape[0], image_np.shape[1], 4))
    img[:, :, 3] = 0  # Transparent
    for idx, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        # Draw borders
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=2)
    overlay = Image.fromarray((img * 255).astype(np.uint8))
    base = Image.fromarray(image_np)
    result = Image.alpha_composite(base.convert('RGBA'), overlay)
    # Add simple segment numbers as labels
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=20)  # Adjust size if needed
    for i, ann in enumerate(sorted_anns):
        bbox = get_bbox_from_mask(ann['segmentation'])
        label = f"#{i+1}"
        text_pos = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        draw.text(text_pos, label, fill=(255, 255, 255, 255), font=font, anchor="mm")
    result.save('overlay.png')

show_anns(masks[:len(segments)], [], image_np)

print("Saved overlay.png")
print(f"\nProcessing complete! Check the '{output_folder}' folder for individual segment images.")
