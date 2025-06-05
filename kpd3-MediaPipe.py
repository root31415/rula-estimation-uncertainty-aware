import cv2
import mediapipe as mp
import numpy as np
from PIL import Image # Import Pillow's Image module for display

# (All other necessary imports like ImageDraw, ImageFont would be here if adding text with Pillow)

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5)

# Define the MediaPipe PoseLandmark indices that correspond to our 17 COCO-style keypoints
COCO17_MEDIAPIPE_LANDMARK_INDICES = [
    mp.solutions.pose.PoseLandmark.NOSE.value, mp.solutions.pose.PoseLandmark.LEFT_EYE.value,
    mp.solutions.pose.PoseLandmark.RIGHT_EYE.value, mp.solutions.pose.PoseLandmark.LEFT_EAR.value,
    mp.solutions.pose.PoseLandmark.RIGHT_EAR.value, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value, mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value, mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value,
]
COCO17_KEYPOINT_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
]

# Skeleton connections for the 17 COCO-style points (using 0-16 indices)
COCO17_SKELETON_GROUPS_FOR_DRAWING = [
    (("L_Shoulder", "L_Eye"), "head_upper_torso"), (("R_Shoulder", "R_Eye"), "head_upper_torso"),
    (("L_Eye", "R_Eye"), "head_upper_torso"),
    (("L_Shoulder", "L_Elbow"), "arms"), (("L_Elbow", "L_Wrist"), "arms"),
    (("R_Shoulder", "R_Elbow"), "arms"), (("R_Elbow", "R_Wrist"), "arms"),
    (("L_Shoulder", "R_Shoulder"), "torso_rect"), (("R_Shoulder", "R_Hip"), "torso_rect"),
    (("R_Hip", "L_Hip"), "torso_rect"), (("L_Hip", "L_Shoulder"), "torso_rect"),
    (("L_Hip", "L_Knee"), "legs"), (("L_Knee", "L_Ankle"), "legs"),
    (("R_Hip", "R_Knee"), "legs"), (("R_Knee", "R_Ankle"), "legs")
]
# Create a mapping from name to index for easier lookup in COCO17_SKELETON_GROUPS_FOR_DRAWING
COCO17_NAME_TO_INDEX = {name: i for i, name in enumerate(COCO17_KEYPOINT_NAMES)}


# Colors for drawing (OpenCV uses BGR format for drawing, then we convert to RGB for Pillow)
SINGLE_KEYPOINT_COLOR_CV = (180, 130, 70)  # BGR: Steel Blue / Teal-ish
LINE_COLORS_BY_GROUP_CV = {
    "head_upper_torso": (0, 255, 0),      # Green (BGR)
    "arms": (255, 0, 0),                  # Blue (BGR)
    "torso_rect": (255, 0, 255),          # Magenta (BGR)
    "legs": (0, 165, 255)                 # Orange (BGR)
}

POINT_RADIUS = 5
LINE_WIDTH = 2
VISIBILITY_THRESHOLD = 0.3

# --- Load Image ---
image_path = "images/5.jpg"  #  <--- REPLACE WITH THE PATH TO YOUR IMAGE
image_cv = cv2.imread(image_path) # Load with OpenCV for MediaPipe processing

if image_cv is None:
    print(f"Error: Could not load image from '{image_path}'")
    exit()

image_height, image_width, _ = image_cv.shape
# We will draw on this OpenCV image (NumPy array) first
annotated_image_cv = image_cv.copy()

# --- Process Image with MediaPipe ---
image_rgb_for_mp = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
image_rgb_for_mp.flags.writeable = False
results = pose.process(image_rgb_for_mp)

# --- Extract and Draw 17 Keypoints and Skeleton using OpenCV drawing functions ---
extracted_coco17_keypoints_map = {} # Store by name: {name: {"x":cx, "y":cy, "score": vis}}

if results.pose_landmarks:
    print("MediaPipe Pose landmarks detected.")
    all_mp_landmarks = results.pose_landmarks.landmark

    for coco_idx, mp_landmark_idx in enumerate(COCO17_MEDIAPIPE_LANDMARK_INDICES):
        mp_lm = all_mp_landmarks[mp_landmark_idx]
        cx = int(mp_lm.x * image_width)
        cy = int(mp_lm.y * image_height)
        visibility = mp_lm.visibility
        keypoint_name = COCO17_KEYPOINT_NAMES[coco_idx]

        if visibility >= VISIBILITY_THRESHOLD:
            extracted_coco17_keypoints_map[keypoint_name] = {
                "x": cx, "y": cy, "score": visibility
            }
            cv2.circle(annotated_image_cv, (cx, cy), POINT_RADIUS, SINGLE_KEYPOINT_COLOR_CV, -1)
    
    lines_drawn = 0
    for (kp_name1, kp_name2), group_tag in COCO17_SKELETON_GROUPS_FOR_DRAWING:
        kp1_data = extracted_coco17_keypoints_map.get(kp_name1)
        kp2_data = extracted_coco17_keypoints_map.get(kp_name2)

        if kp1_data and kp2_data: # Both keypoints are valid
            pt1 = (kp1_data["x"], kp1_data["y"])
            pt2 = (kp2_data["x"], kp2_data["y"])
            line_color = LINE_COLORS_BY_GROUP_CV.get(group_tag, (255, 255, 255)) # Default to white
            cv2.line(annotated_image_cv, pt1, pt2, line_color, LINE_WIDTH)
            lines_drawn +=1
    print(f"Drew {lines_drawn} skeleton lines.")
else:
    print("No pose landmarks detected by MediaPipe.")


# --- Convert final annotated image (OpenCV BGR) to Pillow RGB Image for display ---
# OpenCV drawing functions operate on BGR NumPy arrays. Pillow's show() expects RGB.
final_image_rgb_numpy = cv2.cvtColor(annotated_image_cv, cv2.COLOR_BGR2RGB)
pil_display_image = Image.fromarray(final_image_rgb_numpy)


# --- Stage 3: Display the image using Pillow's show() method ---
# Optional: If you wanted to draw a title directly on the PIL image:
# from PIL import ImageDraw, ImageFont
# draw_pil = ImageDraw.Draw(pil_display_image)
# try:
#     title_font = ImageFont.truetype("arial.ttf", 20)
# except IOError:
#     title_font = ImageFont.load_default()
# title_text = "MediaPipe Pose - 17 Keypoints (YOLO-Style)"
# draw_pil.text((10, 10), title_text, font=title_font, fill=(0,0,0)) # Black text

print("\nDisplaying image using the system's default image viewer (via Pillow)...")
pil_display_image.show()

print("Image display initiated. The script might terminate while the image viewer remains open.")
print("If the image viewer blocks script termination, close the viewer manually.")

# Release MediaPipe Pose resources
pose.close()