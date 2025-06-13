import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from ultralytics import YOLO
import csv
import os

# --- Imports for Hugging Face Transformers ---
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

# --- 1. Centralized Configuration ---
KEYPOINT_SCORE_THRESHOLD = 0.3
COCO17_KEYPOINT_NAMES = ["Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"]
POINT_RADIUS = 5
LINE_WIDTH = 2
SINGLE_KEYPOINT_COLOR = (70, 130, 180) 
LINE_COLORS_BY_GROUP = {"head_upper_torso": (0, 255, 0), "arms": (0, 0, 255), "torso_rect": (255, 0, 255), "legs": (255, 165, 0)}
SKELETON_DEFINITION = [(("L_Shoulder", "L_Eye"), "head_upper_torso"), (("R_Shoulder", "R_Eye"), "head_upper_torso"), (("L_Eye", "R_Eye"), "head_upper_torso"), (("L_Shoulder", "L_Elbow"), "arms"), (("L_Elbow", "L_Wrist"), "arms"), (("R_Shoulder", "R_Elbow"), "arms"), (("R_Elbow", "R_Wrist"), "arms"), (("L_Shoulder", "R_Shoulder"), "torso_rect"), (("R_Shoulder", "R_Hip"), "torso_rect"), (("R_Hip", "L_Hip"), "torso_rect"), (("L_Hip", "L_Shoulder"), "torso_rect"), (("L_Hip", "L_Knee"), "legs"), (("L_Knee", "L_Ankle"), "legs"), (("R_Hip", "R_Knee"), "legs"), (("R_Knee", "R_Ankle"), "legs")]

# --- 2. Helper, RULA, and Mean Skeleton Functions ---
def add_title_to_image(image, title_text):
    draw = ImageDraw.Draw(image)
    font_size = 24
    try: title_font = ImageFont.truetype("arial.ttf", font_size)
    except IOError: title_font = ImageFont.load_default()
    text_position = (10, 10)
    if hasattr(draw, 'textbbox'):
        text_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (image.width - text_width) // 2
        text_position = (text_x, 10)
    draw.text(text_position, title_text, font=title_font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return image

def draw_pose(draw, keypoints_dict):
    for (kp_name1, kp_name2), group_tag in SKELETON_DEFINITION:
        if kp_name1 in keypoints_dict and kp_name2 in keypoints_dict:
            pt1 = (keypoints_dict[kp_name1]['x'], keypoints_dict[kp_name1]['y'])
            pt2 = (keypoints_dict[kp_name2]['x'], keypoints_dict[kp_name2]['y'])
            line_color = LINE_COLORS_BY_GROUP.get(group_tag, (255, 255, 255))
            draw.line([pt1, pt2], fill=line_color, width=LINE_WIDTH)
    for kp_name, kp_info in keypoints_dict.items():
        x, y = kp_info['x'], kp_info['y']
        draw.ellipse([(x - POINT_RADIUS, y - POINT_RADIUS), (x + POINT_RADIUS, y + POINT_RADIUS)], fill=SINGLE_KEYPOINT_COLOR, outline=SINGLE_KEYPOINT_COLOR)

def get_point(person_row, kp_name):
    if not person_row: return None
    x, y, score = person_row.get(f"{kp_name}_x", 0), person_row.get(f"{kp_name}_y", 0), person_row.get(f"{kp_name}_score", 0.0)
    if score > KEYPOINT_SCORE_THRESHOLD:
        return (x, y)
    return None

def calculate_angle(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None: return None
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_detailed_rula_score(person_row):
    scores = {'upper_arm_score': 0, 'lower_arm_score': 0, 'wrist_score': 0, 'wrist_twist_score': 0, 'posture_score_A': 0, 'final_score_A': 0, 'neck_score': 0, 'trunk_score': 0, 'legs_score': 0, 'posture_score_B': 0, 'final_score_B': 0, 'rula_grand_score': 0, 'angle_upper_arm': 0.0, 'angle_lower_arm': 0.0, 'angle_neck': 0.0, 'angle_trunk': 0.0}
    TABLE_A = [[(1,2,2,3), (2,2,3,4), (2,3,3,4)], [(2,3,3,4), (3,3,4,4), (3,4,4,5)], [(3,4,4,5), (4,4,4,5), (4,5,5,6)], [(4,5,5,6), (5,5,6,7), (5,6,6,7)], [(5,6,6,7), (6,6,7,7), (6,7,7,8)], [(6,7,7,8), (7,7,8,8), (7,7,8,8)]]
    TABLE_B = [[(1,3,2,3), (2,3,2,3), (3,4,3,4), (5,5,4,5), (6,6,5,6), (7,7,6,7)], [(2,3,2,3), (2,3,3,4), (4,5,4,5), (5,6,5,6), (6,7,6,7), (7,8,7,8)], [(3,3,3,4), (3,4,3,4), (4,5,4,5), (5,6,5,6), (6,7,6,7), (7,8,7,8)], [(5,5,5,6), (5,6,6,7), (6,7,7,8), (7,8,8,9), (8,9,9,9), (9,9,9,9)]]
    TABLE_C = [[1,2,3,3,4,5,5], [2,2,3,4,4,5,5], [3,3,3,4,4,5,6], [3,3,3,4,5,6,6], [4,4,4,5,6,7,7], [4,4,5,6,6,7,7], [5,5,6,6,7,7,7], [5,5,6,7,7,7,7]]
    def get_arm_scores(side):
        shoulder, elbow, wrist, hip = get_point(person_row, f'{side}_Shoulder'), get_point(person_row, f'{side}_Elbow'), get_point(person_row, f'{side}_Wrist'), get_point(person_row, f'{side}_Hip')
        if not all([shoulder, elbow, hip, wrist]): return None, None, None, None
        upper_arm_angle = calculate_angle((shoulder[0], shoulder[1] - 100), shoulder, elbow)
        lower_arm_angle = calculate_angle(shoulder, elbow, wrist)
        upper_arm_score = 1
        if upper_arm_angle is not None:
            if 20 < upper_arm_angle <= 45: upper_arm_score = 2
            elif 45 < upper_arm_angle <= 90: upper_arm_score = 3
            elif upper_arm_angle > 90: upper_arm_score = 4
        lower_arm_score = 2
        if lower_arm_angle is not None and 60 < lower_arm_angle < 100: lower_arm_score = 1
        return upper_arm_score, lower_arm_score, upper_arm_angle, lower_arm_angle
    upper_arm_score, lower_arm_score, upper_arm_angle, lower_arm_angle = get_arm_scores('R')
    if upper_arm_score is None: upper_arm_score, lower_arm_score, upper_arm_angle, lower_arm_angle = get_arm_scores('L')
    if upper_arm_score is None: return scores
    scores.update({'upper_arm_score': upper_arm_score, 'lower_arm_score': lower_arm_score, 'wrist_score': 1, 'wrist_twist_score': 1, 'angle_upper_arm': upper_arm_angle or 0.0, 'angle_lower_arm': lower_arm_angle or 0.0})
    try: scores['posture_score_A'] = TABLE_A[scores['upper_arm_score']-1][scores['lower_arm_score']-1][scores['wrist_score']-1]
    except (IndexError, TypeError): scores['posture_score_A'] = 0
    scores['final_score_A'] = scores['posture_score_A']
    r_shoulder, l_shoulder, r_hip, l_hip, nose = get_point(person_row, 'R_Shoulder'), get_point(person_row, 'L_Shoulder'), get_point(person_row, 'R_Hip'), get_point(person_row, 'L_Hip'), get_point(person_row, 'Nose')
    neck_ref = None
    if r_shoulder and l_shoulder: neck_ref = ((r_shoulder[0]+l_shoulder[0])/2, (r_shoulder[1]+l_shoulder[1])/2)
    if nose and neck_ref:
        neck_angle = calculate_angle((neck_ref[0], neck_ref[1]-100), neck_ref, nose)
        if neck_angle is not None: scores['neck_score'], scores['angle_neck'] = (1 if neck_angle <= 20 else 2), neck_angle
    if neck_ref and r_hip and l_hip:
        trunk_mid_point = ((r_hip[0]+l_hip[0])/2, (r_hip[1]+l_hip[1])/2)
        trunk_angle = calculate_angle((neck_ref[0], neck_ref[1]-100), neck_ref, trunk_mid_point)
        if trunk_angle is not None:
            if trunk_angle <= 20: scores['trunk_score'] = 1
            elif 20 < trunk_angle <= 60: scores['trunk_score'] = 3
            else: scores['trunk_score'] = 4
            scores['angle_trunk'] = trunk_angle
    scores['legs_score'] = 1
    try:
        if all(s > 0 for s in [scores['neck_score'], scores['trunk_score'], scores['legs_score']]):
            scores['posture_score_B'] = TABLE_B[scores['neck_score']-1][scores['trunk_score']-1][scores['legs_score']-1]
    except (IndexError, TypeError): scores['posture_score_B'] = 0
    scores['final_score_B'] = scores['posture_score_B']
    try:
        if scores['final_score_A'] > 0 and scores['final_score_B'] > 0:
            a_idx, b_idx = min(scores['final_score_A'], 8)-1, min(scores['final_score_B'], 7)-1
            scores['rula_grand_score'] = TABLE_C[b_idx][a_idx]
    except (IndexError, TypeError): scores['rula_grand_score'] = 0
    return scores

def initialize_person_row(detector_name, person_id=0):
    row = {"detector": detector_name, "person_id": person_id}
    for name in COCO17_KEYPOINT_NAMES:
        row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_score"] = 0, 0, 0.0
    return row

def calculate_robust_mean_skeleton(yolo_pose, vitpose_pose, mediapipe_pose):
    print("Calculating Robust Mean Skeleton...")
    mean_pose = initialize_person_row("Mean-Skeleton")
    poses = [p for p in [yolo_pose, vitpose_pose, mediapipe_pose] if p]
    if not poses: return None
    for kp_name in COCO17_KEYPOINT_NAMES:
        detected_points = []
        for pose in poses:
            point = get_point(pose, kp_name)
            if point:
                detected_points.append({"x": pose[f"{kp_name}_x"], "y": pose[f"{kp_name}_y"], "score": pose[f"{kp_name}_score"]})
        
        if not detected_points: continue

        if len(detected_points) <= 2:
            kept_points = detected_points
        else:
            p1, p2, p3 = [np.array([p['x'], p['y']]) for p in detected_points]
            d12, d13, d23 = np.linalg.norm(p1 - p2), np.linalg.norm(p1 - p3), np.linalg.norm(p2 - p3)
            dist_sums = [d12 + d13, d12 + d23, d13 + d23]
            outlier_index = np.argmax(dist_sums)
            kept_points = [p for i, p in enumerate(detected_points) if i != outlier_index]
        
        if kept_points:
            count = len(kept_points)
            mean_pose[f"{kp_name}_x"] = int(sum(p['x'] for p in kept_points) / count)
            mean_pose[f"{kp_name}_y"] = int(sum(p['y'] for p in kept_points) / count)
            mean_pose[f"{kp_name}_score"] = sum(p['score'] for p in kept_points) / count
    return mean_pose

# --- 3. Detector Functions ---
def detect_with_yolo(image_path, model):
    print("Processing with YOLOv8-Pose...")
    try: image = Image.open(image_path).convert("RGB")
    except Exception as e: return None, [], None
    
    results = model(image_path, verbose=False)
    best_person_pose = None
    best_person_box = None
    
    if results and results[0].keypoints and results[0].keypoints.data.nelement() > 0:
        best_person_idx, max_area = -1, -1
        boxes = results[0].boxes.data
        for i, box in enumerate(boxes):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > max_area:
                max_area, best_person_idx = area, i
        
        if best_person_idx != -1:
            person_kps = results[0].keypoints.data[best_person_idx]
            person_row = initialize_person_row("YOLOv8-Pose", best_person_idx)
            for kp_id, kp_data in enumerate(person_kps):
                if kp_id < len(COCO17_KEYPOINT_NAMES):
                    x, y, score = kp_data.tolist()
                    kp_name = COCO17_KEYPOINT_NAMES[kp_id]
                    person_row[f"{kp_name}_x"], person_row[f"{kp_name}_y"], person_row[f"{kp_name}_score"] = int(x), int(y), score
            best_person_pose = person_row
            best_person_box = boxes[best_person_idx][:4].cpu().numpy()

    annotated_image = image.copy()
    if best_person_pose:
        draw = ImageDraw.Draw(annotated_image)
        keypoints_for_drawing = {name: {'x': best_person_pose[f"{name}_x"], 'y': best_person_pose[f"{name}_y"]} for name in COCO17_KEYPOINT_NAMES if get_point(best_person_pose, name)}
        draw_pose(draw, keypoints_for_drawing)
    
    add_title_to_image(annotated_image, "YOLOv8-Pose")
    return annotated_image, [best_person_pose] if best_person_pose else [], best_person_box


def detect_with_vitpose(image_path, person_processor, person_model, pose_processor, pose_model):
    print("Processing with ViTPose...")
    person_data_list = []
    
    device = person_model.device
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"ViTPose - Error loading image: {e}")
        return None, []

    # Person Detection using RTDetr
    image_height, image_width = image.height, image.width
    inputs_person = person_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_person = person_model(**inputs_person)
    
    results_person = person_processor.post_process_object_detection(outputs_person, target_sizes=torch.tensor([(image_height, image_width)]), threshold=0.3)[0]
    person_boxes_xyxy = results_person["boxes"][results_person["labels"] == 0].cpu().numpy()
    
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    if len(person_boxes_xyxy) > 0:
        # Convert xyxy to xywh
        person_boxes_xywh = person_boxes_xyxy.copy()
        person_boxes_xywh[:, 2] = person_boxes_xywh[:, 2] - person_boxes_xywh[:, 0]
        person_boxes_xywh[:, 3] = person_boxes_xywh[:, 3] - person_boxes_xywh[:, 1]
        
        # --- CRITICAL FIX: Wrap the boxes array in a list for the processor ---
        inputs_pose = pose_processor(image, boxes=[person_boxes_xywh], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_pose = pose_model(**inputs_pose)
        
        # The 'boxes' argument here must also be wrapped in a list
        image_pose_result = pose_processor.post_process_pose_estimation(outputs_pose, boxes=[person_boxes_xywh], threshold=0.1)[0]
        
        best_person_pose = None
        max_keypoints = -1
        if image_pose_result:
            for person_pose in image_pose_result:
                count = sum(1 for score in person_pose["scores"] if score.item() >= KEYPOINT_SCORE_THRESHOLD)
                if count > max_keypoints:
                    max_keypoints = count
                    best_person_pose = person_pose
        
        if best_person_pose:
            person_row = initialize_person_row("ViTPose", 0)
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
    
    if not person_data_list:
        add_title_to_image(annotated_image, "ViTPose (No Detections)")
    else:
        add_title_to_image(annotated_image, "ViTPose")
        
    return annotated_image, person_data_list


def detect_with_mediapipe(image_path, pose_model):
    print("Processing with MediaPipe...")
    try: image = Image.open(image_path).convert("RGB")
    except Exception as e: return None, []
    
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    image_np = np.array(image)
    results = pose_model.process(image_np)
    
    person_row = None
    if results.pose_landmarks:
        person_row = initialize_person_row("MediaPipe")
        keypoints_for_drawing = {}
        mp_indices = [mp.solutions.pose.PoseLandmark.NOSE.value, mp.solutions.pose.PoseLandmark.LEFT_EYE.value, mp.solutions.pose.PoseLandmark.RIGHT_EYE.value, mp.solutions.pose.PoseLandmark.LEFT_EAR.value, mp.solutions.pose.PoseLandmark.RIGHT_EAR.value, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value, mp.solutions.pose.PoseLandmark.LEFT_WRIST.value, mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value, mp.solutions.pose.PoseLandmark.LEFT_HIP.value, mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, mp.solutions.pose.PoseLandmark.LEFT_KNEE.value, mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
        
        for coco_idx, mp_idx in enumerate(mp_indices):
            mp_lm = results.pose_landmarks.landmark[mp_idx]
            keypoint_name = COCO17_KEYPOINT_NAMES[coco_idx]
            x, y, v = int(mp_lm.x * image.width), int(mp_lm.y * image.height), mp_lm.visibility
            person_row[f"{keypoint_name}_x"], person_row[f"{keypoint_name}_y"], person_row[f"{keypoint_name}_score"] = x, y, v
            if v >= KEYPOINT_SCORE_THRESHOLD:
                keypoints_for_drawing[keypoint_name] = {'x': x, 'y': y}
                
        draw_pose(draw, keypoints_for_drawing)
        add_title_to_image(annotated_image, "MediaPipe")
    else:
        add_title_to_image(annotated_image, "MediaPipe (No Detections)")
        
    return annotated_image, [person_row] if person_row else []

# --- 4. Main Execution ---
if __name__ == "__main__":
    input_image_path = "images/43.png"
    output_csv_path = "rula_results_detailed.csv"
    
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at '{input_image_path}'")
        exit()

    print("Loading all models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    yolo_model = YOLO("yolov8m-pose.pt")
    mp_pose_model = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    
    # Models needed for the user's ViTPose logic
    person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)
    pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device)
    print("All models loaded.")

    yolo_result_img, yolo_person_rows, yolo_box = detect_with_yolo(input_image_path, yolo_model)
    
    # Using the user's ViTPose logic
    vitpose_result_img, vitpose_person_rows = detect_with_vitpose(
        input_image_path, person_processor, person_model, pose_processor, pose_model
    )
    
    mediapipe_result_img, mediapipe_person_rows = detect_with_mediapipe(input_image_path, mp_pose_model)

    yolo_pose = yolo_person_rows[0] if yolo_person_rows else None
    vitpose_pose = vitpose_person_rows[0] if vitpose_person_rows else None
    mediapipe_pose = mediapipe_person_rows[0] if mediapipe_person_rows else None
    
    mean_pose = calculate_robust_mean_skeleton(yolo_pose, vitpose_pose, mediapipe_pose)
    all_poses = [p for p in [yolo_pose, vitpose_pose, mediapipe_pose, mean_pose] if p]

    if all_poses:
        for pose_data in all_poses:
            rula_scores = calculate_detailed_rula_score(pose_data)
            pose_data.update(rula_scores)
            pose_data['image_path'] = input_image_path
        
        print(f"\nWriting detailed RULA data for {len(all_poses)} poses to {output_csv_path}...")
        rula_score_names = ['upper_arm_score', 'lower_arm_score', 'wrist_score', 'wrist_twist_score', 'posture_score_A', 'final_score_A', 'neck_score', 'trunk_score', 'legs_score', 'posture_score_B', 'final_score_B', 'rula_grand_score', 'angle_upper_arm', 'angle_lower_arm', 'angle_neck', 'angle_trunk']
        fieldnames = ['image_path', 'detector', 'person_id'] + rula_score_names
        for name in COCO17_KEYPOINT_NAMES:
            fieldnames.extend([f'{name}_x', f'{name}_y', f'{name}_score'])
            
        # --- FIXED: Logic to append to the CSV file ---
        file_exists = os.path.isfile(output_csv_path)
        with open(output_csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists or os.path.getsize(output_csv_path) == 0:
                writer.writeheader()
            writer.writerows(all_poses)
        print("CSV writing complete.")

    images_to_display = [yolo_result_img, vitpose_result_img, mediapipe_result_img]
    if mean_pose:
        image = Image.open(input_image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        keypoints_for_drawing = { kp_name: {'x': mean_pose[f"{kp_name}_x"], 'y': mean_pose[f"{kp_name}_y"]} for kp_name in COCO17_KEYPOINT_NAMES if get_point(mean_pose, kp_name) is not None }
        draw_pose(draw, keypoints_for_drawing)
        rula_score = mean_pose.get('rula_grand_score', 'N/A')
        title = f"Mean-Skeleton (RULA: {rula_score})"
        image = add_title_to_image(image, title)
        images_to_display.append(image)

    valid_images = [img for img in images_to_display if img is not None]
    if valid_images:
        max_height = max(img.height for img in valid_images)
        resized_images = [img.resize((int(img.width * max_height / img.height), max_height)) for img in valid_images]
        total_width = sum(img.width for img in resized_images)
        
        composite_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in resized_images:
            composite_image.paste(img, (x_offset, 0))
            x_offset += img.width
        
        print("\nDisplaying composite image...")
        composite_image.show()
    else:
        print("No valid poses were detected to display.")
        
    mp_pose_model.close()