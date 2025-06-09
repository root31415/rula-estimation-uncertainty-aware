import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import mediapipe as mp
from ultralytics import YOLO
import csv
import os

# --- 1. Centralized Configuration for Drawing & Keypoints ---

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

# Shared keypoint names and score threshold
KEYPOINT_SCORE_THRESHOLD = 0.3
COCO17_KEYPOINT_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
]

# Shared visual style for all detectors
POINT_RADIUS = 5
LINE_WIDTH = 2
SINGLE_KEYPOINT_COLOR = (70, 130, 180) # Steel Blue / Teal-ish
LINE_COLORS_BY_GROUP = {
    "head_upper_torso": (0, 255, 0),      # Green
    "arms": (0, 0, 255),                  # Blue
    "torso_rect": (255, 0, 255),          # Magenta
    "legs": (255, 165, 0)                 # Orange
}
SKELETON_DEFINITION = [
    (("L_Shoulder", "L_Eye"), "head_upper_torso"), (("R_Shoulder", "R_Eye"), "head_upper_torso"),
    (("L_Eye", "R_Eye"), "head_upper_torso"),
    (("L_Shoulder", "L_Elbow"), "arms"), (("L_Elbow", "L_Wrist"), "arms"),
    (("R_Shoulder", "R_Elbow"), "arms"), (("R_Elbow", "R_Wrist"), "arms"),
    (("L_Shoulder", "R_Shoulder"), "torso_rect"), (("R_Shoulder", "R_Hip"), "torso_rect"),
    (("R_Hip", "L_Hip"), "torso_rect"), (("L_Hip", "L_Shoulder"), "torso_rect"),
    (("L_Hip", "L_Knee"), "legs"), (("L_Knee", "L_Ankle"), "legs"),
    (("R_Hip", "R_Knee"), "legs"), (("R_Knee", "R_Ankle"), "legs")
]

# --- 2. Helper Functions ---
def add_title_to_image(image, title_text):
    """Draws a title on the top-center of a Pillow Image."""
    draw = ImageDraw.Draw(image)
    font_size = 24
    try:
        title_font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        title_font = ImageFont.load_default()
    
    text_position = (10, 10)
    try:
        text_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (image.width - text_width) // 2
        text_position = (text_x, 10)
    except AttributeError:
        pass

    draw.text(text_position, title_text, font=title_font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return image

def draw_pose(draw, keypoints_dict):
    """Draws a single pose skeleton and points on a Pillow Draw object."""
    # Draw skeleton lines first
    for (kp_name1, kp_name2), group_tag in SKELETON_DEFINITION:
        if kp_name1 in keypoints_dict and kp_name2 in keypoints_dict:
            pt1 = (keypoints_dict[kp_name1]['x'], keypoints_dict[kp_name1]['y'])
            pt2 = (keypoints_dict[kp_name2]['x'], keypoints_dict[kp_name2]['y'])
            line_color = LINE_COLORS_BY_GROUP.get(group_tag, (255, 255, 255))
            draw.line([pt1, pt2], fill=line_color, width=LINE_WIDTH)
    
    # Draw keypoints on top of lines
    for kp_name, kp_info in keypoints_dict.items():
        x, y = kp_info['x'], kp_info['y']
        draw.ellipse([(x - POINT_RADIUS, y - POINT_RADIUS), (x + POINT_RADIUS, y + POINT_RADIUS)],
                     fill=SINGLE_KEYPOINT_COLOR, outline=SINGLE_KEYPOINT_COLOR)

# --- 3. Detector Functions ---

def detect_with_yolo(image_path, model):
    print("Processing with YOLOv8-Pose...")
    person_data_list = []
    
    image = Image.open(image_path).convert("RGB")
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    results = model(image_path)
    best_person_kps = None
    max_keypoints = -1

    for r in results:
        if r.keypoints is not None and r.keypoints.data is not None:
            # Find the person with the most keypoints above the threshold
            for person_kps in r.keypoints.data:
                count = sum(1 for kp in person_kps if kp[2] >= KEYPOINT_SCORE_THRESHOLD)
                if count > max_keypoints:
                    max_keypoints = count
                    best_person_kps = person_kps
    
    # Process and draw only the best person
    if best_person_kps is not None:
        person_row = {"detector": "YOLOv8-Pose", "person_id": 0}
        keypoints_for_drawing = {}
        for keypoint_id, keypoint_data in enumerate(best_person_kps):
            x, y, score = keypoint_data.tolist()
            keypoint_name = COCO17_KEYPOINT_NAMES[keypoint_id]
            person_row[f"{keypoint_name}_x"] = int(x)
            person_row[f"{keypoint_name}_y"] = int(y)
            person_row[f"{keypoint_name}_score"] = score
            if score >= KEYPOINT_SCORE_THRESHOLD:
                keypoints_for_drawing[keypoint_name] = {'x': int(x), 'y': int(y), 'score': score}
        
        person_data_list.append(person_row)
        draw_pose(draw, keypoints_for_drawing)
        annotated_image = add_title_to_image(annotated_image, "YOLOv8-Pose")
    else:
        annotated_image = add_title_to_image(annotated_image, "YOLOv8-Pose (No Detections)")
    
    return annotated_image, person_data_list

def detect_with_vitpose(image_path):
    print("Processing with ViTPose...")
    person_data_list = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"ViTPose - Error loading image: {e}")
        return None, []

    # Person Detection
    person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)
    
    image_height, image_width = image.height, image.width
    inputs_person = person_image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_person = person_model(**inputs_person)
    results_person = person_image_processor.post_process_object_detection(outputs_person, target_sizes=torch.tensor([(image_height, image_width)]), threshold=0.3)[0]
    person_boxes = results_person["boxes"][results_person["labels"] == 0].cpu().numpy()
    
    if person_boxes.shape[0] > 0:
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
    
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    if len(person_boxes) > 0:
        pose_image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)
        
        inputs_pose = pose_image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_pose = pose_model(**inputs_pose)
        image_pose_result = pose_image_processor.post_process_pose_estimation(outputs_pose, boxes=[person_boxes], threshold=0.1)[0]
        
        # Select person with the most keypoints
        best_person_pose = None
        max_keypoints = -1
        if image_pose_result:
            for person_pose in image_pose_result:
                count = sum(1 for score in person_pose["scores"] if score.item() >= KEYPOINT_SCORE_THRESHOLD)
                if count > max_keypoints:
                    max_keypoints = count
                    best_person_pose = person_pose
        
        # Process and draw only the best person
        if best_person_pose:
            person_row = {"detector": "ViTPose", "person_id": 0}
            keypoints_for_drawing = {}
            for keypoint_tensor, label_tensor, score_tensor in zip(best_person_pose["keypoints"], best_person_pose["labels"], best_person_pose["scores"]):
                keypoint_name = pose_model.config.id2label[label_tensor.item()]
                x, y, score = keypoint_tensor[0].item(), keypoint_tensor[1].item(), score_tensor.item()
                
                person_row[f"{keypoint_name}_x"] = int(x)
                person_row[f"{keypoint_name}_y"] = int(y)
                person_row[f"{keypoint_name}_score"] = score
                
                if score >= KEYPOINT_SCORE_THRESHOLD:
                    keypoints_for_drawing[keypoint_name] = {'x': int(x), 'y': int(y), 'score': score}
            
            person_data_list.append(person_row)
            draw_pose(draw, keypoints_for_drawing)
    
    annotated_image = add_title_to_image(annotated_image, "ViTPose")
    return annotated_image, person_data_list

def detect_with_mediapipe(image_path, pose_model):
    print("Processing with MediaPipe...")
    person_data_list = []
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"MediaPipe - Error loading image: {e}")
        return None, []
        
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    image_height, image_width = image.height, image.width
    
    # MediaPipe needs a NumPy array
    image_rgb_for_mp = np.array(image)
    results = pose_model.process(image_rgb_for_mp)

    if results.pose_landmarks:
        # MediaPipe returns only one person, so no selection needed
        COCO17_MEDIAPIPE_LANDMARK_INDICES = [mp.solutions.pose.PoseLandmark.NOSE.value, mp.solutions.pose.PoseLandmark.LEFT_EYE.value, mp.solutions.pose.PoseLandmark.RIGHT_EYE.value, mp.solutions.pose.PoseLandmark.LEFT_EAR.value, mp.solutions.pose.PoseLandmark.RIGHT_EAR.value, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value, mp.solutions.pose.PoseLandmark.LEFT_WRIST.value, mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value, mp.solutions.pose.PoseLandmark.LEFT_HIP.value, mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, mp.solutions.pose.PoseLandmark.LEFT_KNEE.value, mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
        
        person_row = {"detector": "MediaPipe", "person_id": 0}
        keypoints_for_drawing = {}
        for coco_idx, mp_idx in enumerate(COCO17_MEDIAPIPE_LANDMARK_INDICES):
            mp_lm = results.pose_landmarks.landmark[mp_idx]
            keypoint_name = COCO17_KEYPOINT_NAMES[coco_idx]
            x, y, visibility = int(mp_lm.x * image_width), int(mp_lm.y * image_height), mp_lm.visibility
            
            person_row[f"{keypoint_name}_x"] = x
            person_row[f"{keypoint_name}_y"] = y
            person_row[f"{keypoint_name}_score"] = visibility
            
            if visibility >= KEYPOINT_SCORE_THRESHOLD:
                keypoints_for_drawing[keypoint_name] = {'x': x, 'y': y, 'score': visibility}
        
        person_data_list.append(person_row)
        draw_pose(draw, keypoints_for_drawing)
    
    annotated_image = add_title_to_image(annotated_image, "MediaPipe")
    return annotated_image, person_data_list

# --- 4. Main Execution ---
if __name__ == "__main__":
    input_image_path = "images/43.png"
    output_csv_path = "keypoint_results_wide.csv"
    
    print("Loading models...")
    yolo_model = YOLO("yolov8s-pose.pt")
    mp_pose_model = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
    print("Models loaded.")

    yolo_result_img, yolo_person_rows = detect_with_yolo(input_image_path, yolo_model)
    vitpose_result_img, vitpose_person_rows = detect_with_vitpose(input_image_path)
    mediapipe_result_img, mediapipe_person_rows = detect_with_mediapipe(input_image_path, mp_pose_model)

    all_person_rows = []
    for row in yolo_person_rows: row['image_path'] = input_image_path; all_person_rows.append(row)
    for row in vitpose_person_rows: row['image_path'] = input_image_path; all_person_rows.append(row)
    for row in mediapipe_person_rows: row['image_path'] = input_image_path; all_person_rows.append(row)

    if all_person_rows:
        print(f"\nWriting data for {len(all_person_rows)} detected poses to {output_csv_path}...")
        
        fieldnames = ['image_path', 'person_id', 'detector']
        for name in COCO17_KEYPOINT_NAMES:
            fieldnames.extend([f'{name}_x', f'{name}_y', f'{name}_score'])
            
        file_exists = os.path.isfile(output_csv_path)
        with open(output_csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists or os.path.getsize(output_csv_path) == 0:
                writer.writeheader()
            writer.writerows(all_person_rows)
        print("CSV writing complete.")

    images = [yolo_result_img, vitpose_result_img, mediapipe_result_img]
    valid_images = [img for img in images if img is not None]

    if valid_images:
        max_height = max(img.height for img in valid_images)
        resized_images = [img.resize((int(img.width * max_height / img.height), max_height)) for img in valid_images]
        total_width = sum(img.width for img in resized_images)
        composite_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in resized_images:
            composite_image.paste(img, (x_offset, 0))
            x_offset += img.width
        
        print("\nDisplaying composite image with results from all detectors...")
        composite_image.show()
    else:
        print("All detectors failed to process the image.")
        
    mp_pose_model.close()