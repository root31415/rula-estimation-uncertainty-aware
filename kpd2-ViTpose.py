import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Added ImageFont for optional title
# import matplotlib.pyplot as plt # Matplotlib is no longer used for display

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------
# Load local image using Pillow
# ------------------------------------------------------------------------
local_image_path = "images/8.jpg"  # <--- Make sure this path is correct
try:
    image = Image.open(local_image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: Could not load image from '{local_image_path}'")
    print("Please check the file path.")
    exit()
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# ------------------------------------------------------------------------
# Stage 1. Detect humans on the image
# ------------------------------------------------------------------------
person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

image_height, image_width = image.height, image.width # Still useful for potential text placement
inputs_person = person_image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs_person = person_model(**inputs_person)

results_person = person_image_processor.post_process_object_detection(
    outputs_person, target_sizes=torch.tensor([(image_height, image_width)]), threshold=0.3
)
result_person = results_person[0]

person_boxes = result_person["boxes"][result_person["labels"] == 0]
person_boxes = person_boxes.cpu().numpy()

if person_boxes.shape[0] > 0:
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

# ------------------------------------------------------------------------
# Stage 2. Detect keypoints and draw pose for each person found
# ------------------------------------------------------------------------
pose_image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

draw_image = image.copy()
draw = ImageDraw.Draw(draw_image) # The 'draw' object is used for all drawing, including optional text

POINT_RADIUS = 5
LINE_WIDTH = 2
KEYPOINT_SCORE_THRESHOLD = 0.05

SINGLE_KEYPOINT_COLOR = (70, 130, 180) # Steel Blue / Teal-ish

SKELETON_DEFINITION = [
    (("L_Shoulder", "L_Eye"), "head_upper_torso"),
    (("R_Shoulder", "R_Eye"), "head_upper_torso"),
    (("L_Eye", "R_Eye"), "head_upper_torso"),
    (("L_Shoulder", "L_Elbow"), "arms"), (("L_Elbow", "L_Wrist"), "arms"),
    (("R_Shoulder", "R_Elbow"), "arms"), (("R_Elbow", "R_Wrist"), "arms"),
    (("L_Shoulder", "R_Shoulder"), "torso_rect"),
    (("R_Shoulder", "R_Hip"), "torso_rect"),
    (("R_Hip", "L_Hip"), "torso_rect"),
    (("L_Hip", "L_Shoulder"), "torso_rect"),
    (("L_Hip", "L_Knee"), "legs"), (("L_Knee", "L_Ankle"), "legs"),
    (("R_Hip", "R_Knee"), "legs"), (("R_Knee", "R_Ankle"), "legs"),
]

LINE_COLORS_BY_GROUP = {
    "head_upper_torso": (0, 255, 0),
    "arms": (0, 0, 255),
    "torso_rect": (255, 0, 255),
    "legs": (255, 165, 0)
}

if len(person_boxes) > 0:
    inputs_pose = pose_image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs_pose = pose_model(**inputs_pose)

    pose_results = pose_image_processor.post_process_pose_estimation(
        outputs_pose, boxes=[person_boxes], threshold=0.1
    )
    image_pose_result = pose_results[0]

    print(f"\nFound {len(image_pose_result)} person(s) for keypoint estimation.")

    for i, person_pose in enumerate(image_pose_result):
        print(f"\n--- Processing Person #{i} ---")
        visible_keypoints_coords = {}

        for keypoint_tensor, label_tensor, score_tensor in zip(
            person_pose["keypoints"], person_pose["labels"], person_pose["scores"]
        ):
            keypoint_name = pose_model.config.id2label[label_tensor.item()]
            x, y = keypoint_tensor[0].item(), keypoint_tensor[1].item()
            score = score_tensor.item()
            
            # print(f"  Keypoint: {keypoint_name:<15} Score: {score:.2f}", end="") # Less verbose for routine runs

            if score >= KEYPOINT_SCORE_THRESHOLD:
                # print(" (Above Threshold - Drawing Point)") # Less verbose
                ellipse_bbox = [
                    (x - POINT_RADIUS, y - POINT_RADIUS),
                    (x + POINT_RADIUS, y + POINT_RADIUS)
                ]
                draw.ellipse(ellipse_bbox, fill=SINGLE_KEYPOINT_COLOR, outline=SINGLE_KEYPOINT_COLOR)
                visible_keypoints_coords[keypoint_name] = (x, y)
            # else:
                # print(" (Below Threshold)") # Less verbose
        
        # print(f"\n  Person #{i} - Keypoints eligible for lines (met threshold {KEYPOINT_SCORE_THRESHOLD}):") # Less verbose
        # print(f"  {sorted(list(visible_keypoints_coords.keys()))}") # Less verbose
        # print(f"\n  Attempting to draw {len(SKELETON_DEFINITION)} defined skeleton connections for Person #{i}...") # Less verbose
        
        lines_drawn_for_person = 0
        for (kp_name1, kp_name2), group_tag in SKELETON_DEFINITION:
            kp1_present = kp_name1 in visible_keypoints_coords
            kp2_present = kp_name2 in visible_keypoints_coords
            
            if kp1_present and kp2_present:
                x1, y1 = visible_keypoints_coords[kp_name1]
                x2, y2 = visible_keypoints_coords[kp_name2]
                line_color = LINE_COLORS_BY_GROUP.get(group_tag, (255,255,255))
                draw.line([(x1, y1), (x2, y2)], fill=line_color, width=LINE_WIDTH)
                lines_drawn_for_person += 1
        
        print(f"  Person #{i} - Drew {lines_drawn_for_person} lines.")
        # print(f"--- Finished Processing Person #{i} ---\n") # Less verbose
else:
    print("No persons detected by the person detector.")

# ------------------------------------------------------------------------
# Stage 3. Display the image using Pillow's show() method
# ------------------------------------------------------------------------

# --- Optional: Add title text directly onto the image ---
# You'll need a font file (e.g., "arial.ttf").
# If you don't have a specific font, Pillow uses a small default one.
# title_text = f"ViTPose - YOLO-Style Pose (KP Thr: {KEYPOINT_SCORE_THRESHOLD})"
# text_position = (10, 10)  # Pixels from top-left corner
# text_color_rgb = (0, 0, 0)   # Black
#
# try:
#     # Try to load a specific font and size.
#     # You might need to provide the full path to a .ttf font file on your system.
#     # Common fonts: "arial.ttf" (Windows), "DejaVuSans.ttf" (Linux), "Arial.ttf" (macOS)
#     title_font = ImageFont.truetype("arial.ttf", 20)
# except IOError:
#     print("Arial font not found. Using Pillow's default font for title.")
#     title_font = ImageFont.load_default() # Fallback to Pillow's basic default font
#
# # Get text size to potentially center it or position it better
# # For Pillow versions that support it (usually >9.2.0 for anchor, older versions use textsize)
# try:
#    text_bbox = draw.textbbox(text_position, title_text, font=title_font) # For text length
#    # Example: position at top center
#    # text_x = (image_width - (text_bbox[2] - text_bbox[0])) // 2
#    # text_position = (text_x, 10)
# except AttributeError: # Older Pillow versions might not have textbbox
#    pass 
#
# draw.text(text_position, title_text, font=title_font, fill=text_color_rgb)
# --- End of Optional Title ---

# Display the image using the system's default image viewer
draw_image.show()

print("\nImage displayed. The script might terminate while the image viewer remains open.")
print("If the image viewer blocks the script, close the viewer to allow the script to fully exit if needed.")