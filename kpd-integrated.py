import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import mediapipe as mp
from ultralytics import YOLO

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

# --- Helper Function to Add Titles to Images ---
def add_title_to_image(image, title_text):
    """Draws a title on the top-center of a Pillow Image."""
    draw = ImageDraw.Draw(image)
    font_size = 24
    try:
        # Use a common, recognizable font. Adjust path if needed.
        title_font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback to Pillow's basic default font if Arial isn't found
        title_font = ImageFont.load_default()

    # Calculate text position for centering
    text_position = (10, 10) # Default
    try:
        # Get bounding box of text to center it
        text_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (image.width - text_width) // 2
        text_position = (text_x, 10)
    except AttributeError:
        # Older Pillow versions might not have textbbox, stick to default
        pass

    # Draw text with a black stroke for better visibility
    draw.text(text_position, title_text, font=title_font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return image


# --- Detector 1: YOLOv8-Pose ---
def detect_with_yolo(image_path, model):
    """Processes an image with YOLOv8-Pose and returns the annotated Pillow Image."""
    print("Processing with YOLOv8-Pose...")
    results = model(image_path)
    annotated_image = None
    
    for r in results:
        # Use the plot() method to get the annotated image as a NumPy array (BGR)
        im_array = r.plot(boxes=False) # Hides bounding boxes
        # Convert BGR NumPy array to RGB Pillow Image
        annotated_image = Image.fromarray(im_array[..., ::-1])

    if annotated_image:
        annotated_image = add_title_to_image(annotated_image, "YOLOv8-Pose")
    else: # If no detection, return the original image
        annotated_image = Image.open(image_path).convert("RGB")
        annotated_image = add_title_to_image(annotated_image, "YOLOv8-Pose (No Detections)")

    return annotated_image

# --- Detector 2: ViTPose ---
def detect_with_vitpose(image_path):
    """Processes an image with ViTPose and returns the annotated Pillow Image."""
    print("Processing with ViTPose...")
    # This function encapsulates the logic from kpd2-ViTpose.py
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"ViTPose - Error loading image: {e}")
        return None

    # Stage 1: Person Detection
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
    
    # Stage 2: Pose Estimation
    draw_image = image.copy()
    if len(person_boxes) > 0:
        pose_image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)
        
        SKELETON_DEFINITION = [
            (("L_Shoulder", "L_Eye"), "head_upper_torso"), (("R_Shoulder", "R_Eye"), "head_upper_torso"),
            (("L_Eye", "R_Eye"), "head_upper_torso"),
            (("L_Shoulder", "L_Elbow"), "arms"), (("L_Elbow", "L_Wrist"), "arms"),
            (("R_Shoulder", "R_Elbow"), "arms"), (("R_Elbow", "R_Wrist"), "arms"),
            (("L_Shoulder", "R_Shoulder"), "torso_rect"), (("R_Shoulder", "R_Hip"), "torso_rect"),
            (("R_Hip", "L_Hip"), "torso_rect"), (("L_Hip", "L_Shoulder"), "torso_rect"),
            (("L_Hip", "L_Knee"), "legs"), (("L_Knee", "L_Ankle"), "legs"),
            (("R_Hip", "R_Knee"), "legs"), (("R_Knee", "R_Ankle"), "legs"),
        ]
        LINE_COLORS_BY_GROUP = {"head_upper_torso": (0, 255, 0), "arms": (0, 0, 255), "torso_rect": (255, 0, 255), "legs": (255, 165, 0)}
        SINGLE_KEYPOINT_COLOR = (70, 130, 180)
        
        inputs_pose = pose_image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_pose = pose_model(**inputs_pose)
        image_pose_result = pose_image_processor.post_process_pose_estimation(outputs_pose, boxes=[person_boxes], threshold=0.1)[0]
        
        draw = ImageDraw.Draw(draw_image)
        for person_pose in image_pose_result:
            visible_keypoints_coords = {}
            for keypoint_tensor, label_tensor, score_tensor in zip(person_pose["keypoints"], person_pose["labels"], person_pose["scores"]):
                if score_tensor.item() >= 0.05:
                    keypoint_name = pose_model.config.id2label[label_tensor.item()]
                    x, y = keypoint_tensor[0].item(), keypoint_tensor[1].item()
                    draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill=SINGLE_KEYPOINT_COLOR)
                    visible_keypoints_coords[keypoint_name] = (x, y)
            for (kp_name1, kp_name2), group_tag in SKELETON_DEFINITION:
                if kp_name1 in visible_keypoints_coords and kp_name2 in visible_keypoints_coords:
                    x1, y1 = visible_keypoints_coords[kp_name1]
                    x2, y2 = visible_keypoints_coords[kp_name2]
                    draw.line([(x1, y1), (x2, y2)], fill=LINE_COLORS_BY_GROUP.get(group_tag), width=2)
    
    draw_image = add_title_to_image(draw_image, "ViTPose")
    return draw_image


# --- Detector 3: MediaPipe ---
def detect_with_mediapipe(image_path, pose_model):
    """Processes an image with MediaPipe Pose and returns the annotated Pillow Image."""
    print("Processing with MediaPipe...")
    # This function encapsulates the logic from kpd3-MediaPipe.py
    
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"MediaPipe - Error loading image: {image_path}")
        return None
        
    annotated_image_cv = image_cv.copy()
    image_height, image_width, _ = image_cv.shape
    
    # Process image
    image_rgb_for_mp = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb_for_mp)

    # Drawing logic
    if results.pose_landmarks:
        COCO17_MEDIAPIPE_LANDMARK_INDICES = [mp.solutions.pose.PoseLandmark.NOSE.value, mp.solutions.pose.PoseLandmark.LEFT_EYE.value, mp.solutions.pose.PoseLandmark.RIGHT_EYE.value, mp.solutions.pose.PoseLandmark.LEFT_EAR.value, mp.solutions.pose.PoseLandmark.RIGHT_EAR.value, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value, mp.solutions.pose.PoseLandmark.LEFT_WRIST.value, mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value, mp.solutions.pose.PoseLandmark.LEFT_HIP.value, mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, mp.solutions.pose.PoseLandmark.LEFT_KNEE.value, mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value,]
        COCO17_KEYPOINT_NAMES = ["Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"]
        COCO17_SKELETON_GROUPS_FOR_DRAWING = [
            (("L_Shoulder", "L_Eye"), "head_upper_torso"), (("R_Shoulder", "R_Eye"), "head_upper_torso"), (("L_Eye", "R_Eye"), "head_upper_torso"),
            (("L_Shoulder", "L_Elbow"), "arms"), (("L_Elbow", "L_Wrist"), "arms"), (("R_Shoulder", "R_Elbow"), "arms"), (("R_Elbow", "R_Wrist"), "arms"),
            (("L_Shoulder", "R_Shoulder"), "torso_rect"), (("R_Shoulder", "R_Hip"), "torso_rect"), (("R_Hip", "L_Hip"), "torso_rect"), (("L_Hip", "L_Shoulder"), "torso_rect"),
            (("L_Hip", "L_Knee"), "legs"), (("L_Knee", "L_Ankle"), "legs"), (("R_Hip", "R_Knee"), "legs"), (("R_Knee", "R_Ankle"), "legs")]
        LINE_COLORS_BY_GROUP_CV = {"head_upper_torso": (0, 255, 0), "arms": (255, 0, 0), "torso_rect": (255, 0, 255), "legs": (0, 165, 255)}
        SINGLE_KEYPOINT_COLOR_CV = (180, 130, 70)
        
        extracted_kps = {}
        for coco_idx, mp_idx in enumerate(COCO17_MEDIAPIPE_LANDMARK_INDICES):
            mp_lm = results.pose_landmarks.landmark[mp_idx]
            if mp_lm.visibility >= 0.3:
                keypoint_name = COCO17_KEYPOINT_NAMES[coco_idx]
                cx, cy = int(mp_lm.x * image_width), int(mp_lm.y * image_height)
                extracted_kps[keypoint_name] = {"x": cx, "y": cy}
                cv2.circle(annotated_image_cv, (cx, cy), 5, SINGLE_KEYPOINT_COLOR_CV, -1)
        
        for (kp1_name, kp2_name), group in COCO17_SKELETON_GROUPS_FOR_DRAWING:
            if kp1_name in extracted_kps and kp2_name in extracted_kps:
                pt1 = (extracted_kps[kp1_name]["x"], extracted_kps[kp1_name]["y"])
                pt2 = (extracted_kps[kp2_name]["x"], extracted_kps[kp2_name]["y"])
                cv2.line(annotated_image_cv, pt1, pt2, LINE_COLORS_BY_GROUP_CV.get(group), 2)
    
    # Convert final BGR OpenCV image to RGB Pillow Image
    final_image_pil = Image.fromarray(cv2.cvtColor(annotated_image_cv, cv2.COLOR_BGR2RGB))
    final_image_pil = add_title_to_image(final_image_pil, "MediaPipe")
    return final_image_pil

# --- Main Execution ---
if __name__ == "__main__":
    # Define the single input image for all detectors
    input_image_path = "images/8.jpg"  #  <--- SET YOUR INPUT IMAGE PATH HERE
    
    # --- Load models once ---
    print("Loading models...")
    yolo_model = YOLO("yolov8s-pose.pt")
    mp_pose_model = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
    print("Models loaded.")

    # --- Run detectors ---
    yolo_result_img = detect_with_yolo(input_image_path, yolo_model)
    vitpose_result_img = detect_with_vitpose(input_image_path)
    mediapipe_result_img = detect_with_mediapipe(input_image_path, mp_pose_model)

    # --- Create a composite image to display results side-by-side ---
    images = [yolo_result_img, vitpose_result_img, mediapipe_result_img]
    
    # Filter out any images that failed to process
    valid_images = [img for img in images if img is not None]

    if not valid_images:
        print("All detectors failed to process the image.")
    else:
        # Resize images to the same height for clean concatenation
        max_height = max(img.height for img in valid_images)
        resized_images = []
        for img in valid_images:
            aspect_ratio = img.width / img.height
            new_width = int(aspect_ratio * max_height)
            resized_images.append(img.resize((new_width, max_height)))

        total_width = sum(img.width for img in resized_images)
        
        # Create a new blank image
        composite_image = Image.new('RGB', (total_width, max_height))
        
        # Paste each result image into the composite image
        x_offset = 0
        for img in resized_images:
            composite_image.paste(img, (x_offset, 0))
            x_offset += img.width
        
        # Display the final composite image
        print("\nDisplaying composite image with results from all detectors...")
        composite_image.show()
        # To save the composite image:
        # composite_image.save("composite_pose_results.jpg")
        
    # Clean up MediaPipe model
    mp_pose_model.close()