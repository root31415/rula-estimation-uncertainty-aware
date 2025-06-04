import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import math
import sys

# --- Constants for Keypoint Indices (COCO 17 format) ---
KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16

# --- Helper Functions for Geometry ---
def get_keypoint_np(kp_data):
    if kp_data is None: return None
    if hasattr(kp_data, 'cpu') and hasattr(kp_data, 'numpy'):
        return kp_data.cpu().numpy()
    return np.array(kp_data)

def get_keypoint(kpts_array, index):
    kp_value = kpts_array[index] if kpts_array is not None and index < len(kpts_array) else None
    if kp_value is not None and kp_value.sum() == 0: return None
    return kp_value

def calculate_angle_2d(p1_data, p2_data, p3_data): # Calculates angle at p2
    p1, p2, p3 = get_keypoint_np(p1_data), get_keypoint_np(p2_data), get_keypoint_np(p3_data)
    if p1 is None or p2 is None or p3 is None: return None
    v1, v2 = p1 - p2, p3 - p2
    dot_product = np.dot(v1, v2)
    mag_v1, mag_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if mag_v1 == 0 or mag_v2 == 0: return None
    angle_cos = np.clip(dot_product / (mag_v1 * mag_v2), -1.0, 1.0)
    return np.degrees(np.arccos(angle_cos))

def get_midpoint(p1_data, p2_data):
    p1, p2 = get_keypoint_np(p1_data), get_keypoint_np(p2_data)
    if p1 is None or p2 is None: return None
    return np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])

def get_vector_angle_with_vertical_y_down(p_start_data, p_end_data):
    # Angle with positive Y axis (downwards). Positive for flexion, negative for extension.
    p_start, p_end = get_keypoint_np(p_start_data), get_keypoint_np(p_end_data)
    if p_start is None or p_end is None: return None
    vector = p_end - p_start
    if np.linalg.norm(vector) == 0: return 0
    vertical_ref = np.array([0, 1]) # Y positive is downwards
    unit_vector = vector / np.linalg.norm(vector)
    dot_product = np.dot(unit_vector, vertical_ref)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    # Sign based on y-component of the vector itself (more direct for flexion/extension)
    if vector[1] > 0: return angle_deg    # Flexion (end point lower than start point)
    elif vector[1] < 0: return -angle_deg # Extension (end point higher than start point)
    else: return 0 # Horizontal

def get_vector_angle_with_horizontal_x_right(p_start_data, p_end_data):
    p_start, p_end = get_keypoint_np(p_start_data), get_keypoint_np(p_end_data)
    if p_start is None or p_end is None: return None
    vector = p_end - p_start
    if np.linalg.norm(vector) == 0: return 0
    horizontal_ref = np.array([1, 0]) # X positive is to the right
    unit_vector = vector / np.linalg.norm(vector)
    dot_product = np.dot(unit_vector, horizontal_ref)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    # Sign based on y-component for typical screen coordinates (up is negative y)
    if vector[1] < 0: return angle_deg # Points upwards
    else: return -angle_deg # Points downwards or horizontal

# --- RULA Scoring Logic ---
def score_upper_arm(r_shoulder_pt, r_elbow_pt, l_shoulder_pt, r_hip_pt, l_hip_pt, img_width_for_ref=None):
    # Base flexion/extension score
    mid_hip = get_midpoint(l_hip_pt, r_hip_pt)
    # Use mid-shoulder of analyzed arm side for trunk reference if hip data is poor or for specific analysis
    trunk_ref_pt_upper = get_keypoint_np(r_shoulder_pt) # Or mid_shoulder if using that as fixed ref
    
    ua_flex_ext_angle_deg = 0
    if mid_hip is not None and trunk_ref_pt_upper is not None and get_keypoint_np(r_elbow_pt) is not None:
        vec_trunk_approx = trunk_ref_pt_upper - mid_hip
        vec_upper_arm = get_keypoint_np(r_elbow_pt) - trunk_ref_pt_upper
        if np.linalg.norm(vec_trunk_approx) > 0 and np.linalg.norm(vec_upper_arm) > 0:
            dot = np.dot(vec_upper_arm, vec_trunk_approx)
            norm_prod = np.linalg.norm(vec_upper_arm) * np.linalg.norm(vec_trunk_approx)
            angle_rad_rel_trunk = np.arccos(np.clip(dot / norm_prod, -1.0, 1.0))
            ua_flex_ext_angle_deg = np.degrees(angle_rad_rel_trunk)
            # Heuristic sign for flexion (forward) vs extension (backward)
            # If elbow is horizontally in front of shoulder (assuming side view to some extent)
            if get_keypoint_np(r_elbow_pt)[0] < trunk_ref_pt_upper[0]: # Assumes person faces right
                 pass # Positive flexion
            elif get_keypoint_np(r_elbow_pt)[0] == trunk_ref_pt_upper[0]:
                 ua_flex_ext_angle_deg = 0 # Arm straight down along trunk line
            else: # Extension
                 ua_flex_ext_angle_deg = -ua_flex_ext_angle_deg
    else: # Fallback to angle with vertical if trunk line is problematic
        ua_flex_ext_angle_deg = get_vector_angle_with_vertical_y_down(r_shoulder_pt, r_elbow_pt)
        if ua_flex_ext_angle_deg is None: ua_flex_ext_angle_deg = 0

    score = 1
    if 20 < ua_flex_ext_angle_deg <= 45: score = 2
    elif 45 < ua_flex_ext_angle_deg <= 90: score = 3
    elif ua_flex_ext_angle_deg > 90: score = 4
    elif ua_flex_ext_angle_deg < -20: score = 2 # Extension
    elif -20 <= ua_flex_ext_angle_deg < 0 : score = 2 # RULA often groups this with 0-20 flex

    # Adjustments
    adj = 0
    # Naive Abduction: if elbow is far to the side of shoulder
    r_s_np, r_e_np = get_keypoint_np(r_shoulder_pt), get_keypoint_np(r_elbow_pt)
    if r_s_np is not None and r_e_np is not None and img_width_for_ref is not None:
        horizontal_dist = abs(r_s_np[0] - r_e_np[0])
        vertical_dist_arm = abs(r_s_np[1] - r_e_np[1])
        # If horizontal distance is significant compared to vertical (suggests arm out to side)
        # and arm not just hanging down
        if horizontal_dist > vertical_dist_arm * 0.5 and horizontal_dist > img_width_for_ref * 0.05 : # Arm out & significant
            adj += 1 # Abducted
    
    # Naive Shoulder Raised: if analyzed shoulder is much higher than other shoulder
    l_s_np = get_keypoint_np(l_shoulder_pt)
    if r_s_np is not None and l_s_np is not None and img_width_for_ref is not None: # Using img_width for a rough pixel threshold
        if (l_s_np[1] - r_s_np[1]) > img_width_for_ref * 0.03 : # Right shoulder raised (Y is inverted)
            adj +=1
    # Arm supported: -1 (cannot determine from image alone, assume 0)
    return score, adj

def score_lower_arm(r_shoulder_pt, r_elbow_pt, r_wrist_pt):
    elbow_angle = calculate_angle_2d(r_shoulder_pt, r_elbow_pt, r_wrist_pt)
    if elbow_angle is None: return 1, 0
    # RULA optimal: 60-100 degrees. This angle is inner elbow angle.
    score = 1
    if not (60 <= elbow_angle <= 100): score = 2
    # Adjustment for arm working across midline/out to side - simplified to 0
    return score, 0

def score_wrist(r_elbow_pt, r_wrist_pt, r_hand_knuckle_approx=None, img_width_for_ref=None):
    adj_deviation = 0
    # Base score for flexion/extension
    if r_hand_knuckle_approx is None: # No hand point
        score = 2 # Assume mild flexion/extension (non-neutral)
    else:
        # Angle is wrist flexion from straight forearm line
        # angle p_elbow-p_wrist-p_hand. If 180, wrist straight.
        wrist_flex_ext_raw_angle = calculate_angle_2d(r_elbow_pt, r_wrist_pt, r_hand_knuckle_approx)
        if wrist_flex_ext_raw_angle is None:
            wrist_angle_deg = 0 # Assume neutral if calc fails
            score = 1
        else:
            wrist_angle_deg = abs(180 - wrist_flex_ext_raw_angle) # Deviation from straight
            if wrist_angle_deg <= 5 : score = 1
            elif wrist_angle_deg <= 15: score = 2
            else: score = 3
    
    # Naive Ulnar/Radial Deviation: If wrist X is significantly offset from elbow X when arm is vertical-ish
    r_e_np, r_w_np = get_keypoint_np(r_elbow_pt), get_keypoint_np(r_wrist_pt)
    if r_e_np is not None and r_w_np is not None and img_width_for_ref is not None:
        # Check if forearm is roughly vertical (not flexed too much horizontally)
        if abs(r_e_np[0] - r_w_np[0]) < abs(r_e_np[1] - r_w_np[1]) * 0.5: # Forearm more vertical than horizontal
            # Check if wrist x deviates significantly from elbow x (RULA deviation is >15 deg)
            # This is a very rough proxy using pixel offset.
            if abs(r_e_np[0] - r_w_np[0]) > img_width_for_ref * 0.02: # e.g. > 2% of image width
                adj_deviation = 1
    return score, adj_deviation

def score_wrist_twist():
    return 1 # Hardcoded: Assume mid-range pronation/supination (Score 1) or non-markedly twisted

def score_neck_v2(kp_nose_data, kp_l_shoulder_data, kp_r_shoulder_data, kp_l_ear_data, kp_r_ear_data):
    mid_shoulder = get_midpoint(kp_l_shoulder_data, kp_r_shoulder_data)
    head_pt_ears = get_midpoint(kp_l_ear_data, kp_r_ear_data)
    head_pt = head_pt_ears if head_pt_ears is not None else get_keypoint_np(kp_nose_data)

    if head_pt is None or mid_shoulder is None: return 1, 0
    neck_angle_deg = get_vector_angle_with_vertical_y_down(mid_shoulder, head_pt)
    if neck_angle_deg is None: neck_angle_deg = 0

    score = 1 # Flexion 0 to 10 degrees
    if neck_angle_deg < -10: score = 4       # Extension > 10 deg (RULA: in extension)
    elif neck_angle_deg < -0: score = 2      # Slight Extension 0 to -10 deg (RULA: use 0-20 flexion if mapping) -> lets map to score 2 like 10-20 flex
    elif neck_angle_deg > 20: score = 3     # Flexion > 20 deg
    elif neck_angle_deg > 10: score = 2     # Flexion 10-20 deg
    
    adj = 0
    # Naive Neck Twist: If ear midpoint is significantly offset horizontally from nose, and shoulders are somewhat level
    # Naive Neck Side-Bend: If ear-line is tilted significantly relative to shoulder-line
    # These are very complex and error-prone with 2D, keeping adj=0 for now.
    return score, adj

def score_trunk(kp_l_shoulder_data, kp_r_shoulder_data, kp_l_hip_data, kp_r_hip_data):
    mid_shoulder = get_midpoint(kp_l_shoulder_data, kp_r_shoulder_data)
    mid_hip = get_midpoint(kp_l_hip_data, kp_r_hip_data)
    if mid_shoulder is None or mid_hip is None: return 1, 0

    trunk_angle_deg = get_vector_angle_with_vertical_y_down(mid_hip, mid_shoulder)
    if trunk_angle_deg is None: trunk_angle_deg = 0
    
    score = 1 # Upright (0-10 deg flexion)
    if trunk_angle_deg < -10 : # Extension
        score = 2 # RULA: Extension is usually +1 or +2. Simplified to Score 2 or 3
        if trunk_angle_deg < -20 : score = 3 # More severe extension
    elif trunk_angle_deg > 60: score = 4
    elif trunk_angle_deg > 20: score = 3
    elif trunk_angle_deg > 10: score = 2
    
    adj = 0
    # Naive Trunk Twist: Compare shoulder line angle with hip line angle
    # Naive Trunk Side-Bend: If mid_shoulder is significantly offset horizontally from mid_hip AND trunk is relatively upright
    # Keeping adj=0 due to complexity and unreliability in 2D.
    return score, adj

def score_legs(kp_l_knee, kp_r_knee, kp_l_ankle, kp_r_ankle):
    # RULA: 1 if legs/feet supported, balanced. 2 if not.
    # Improved: Check if knees are at very different heights or very wide stance (unstable)
    l_k, r_k = get_keypoint_np(kp_l_knee), get_keypoint_np(kp_r_knee)
    l_a, r_a = get_keypoint_np(kp_l_ankle), get_keypoint_np(kp_r_ankle)

    if l_k is None or r_k is None or l_a is None or r_a is None: return 2 # Missing points implies instability

    # Check for significant asymmetry or wide stance (very rough)
    y_diff_knees = abs(l_k[1] - r_k[1])
    x_dist_ankles = abs(l_a[0] - r_a[0])
    # Assuming a reference like shoulder width if available, else use a heuristic
    # For now, just defaulting to 1 if all points exist, as robust check is hard.
    # If y_diff_knees > (abs(l_k[0]-r_k[0]) * 0.3) or x_dist_ankles > (abs(l_k[0]-r_k[0]) * 1.5):
    #    return 2 # Unstable
    return 1 # Simplified: if all points exist, assume supported for now.

# --- RULA Tables (Unchanged) ---
TABLE_A = [[[[1,2],[2,2]],[[2,2],[2,3]],[[2,3],[3,3]],[[3,3],[3,4]]], [[[2,2],[2,3]],[[3,3],[3,3]],[[3,3],[3,4]],[[4,4],[4,4]]], [[[3,3],[3,3]],[[3,3],[3,4]],[[3,4],[4,4]],[[4,4],[4,5]]], [[[3,4],[4,4]],[[4,4],[4,4]],[[4,4],[4,5]],[[5,5],[5,5]]], [[[4,4],[4,4]],[[4,4],[4,5]],[[4,5],[5,5]],[[5,5],[5,6]]], [[[4,4],[4,5]],[[4,5],[5,5]],[[5,5],[5,6]],[[5,6],[6,7]]]]
TABLE_B = [[[1,3],[2,3],[3,4],[4,5],[5,5],[6,6],[7,7]], [[2,3],[2,3],[3,4],[4,5],[5,5],[6,7],[7,7]], [[3,3],[3,4],[4,4],[4,5],[5,6],[6,7],[7,7]], [[3,4],[4,4],[4,5],[5,5],[6,6],[7,7],[7,7]], [[4,4],[4,5],[5,5],[5,6],[6,7],[7,7],[7,7]], [[4,5],[5,5],[5,6],[6,6],[6,7],[7,7],[7,7]]]
TABLE_C = [[1,2,3,3,4,5,5], [2,2,3,4,4,5,5], [3,3,3,4,4,5,6], [3,3,4,4,5,6,6], [4,4,4,5,6,6,7], [4,4,5,5,6,7,7], [5,5,6,6,7,7,7], [5,5,6,7,7,7,7]]

def get_rula_score(kpts_pixels, img_width, arm_muscle_use_score=0, arm_force_load_score=0, ntl_muscle_use_score=0, ntl_force_load_score=0):
    if kpts_pixels is None or len(kpts_pixels) < 17: return None, "Not enough keypoints."
    kp = lambda i: get_keypoint(kpts_pixels, i)

    ua_s_b, ua_adj = score_upper_arm(kp(KP_R_SHOULDER), kp(KP_R_ELBOW), kp(KP_L_SHOULDER), kp(KP_R_HIP), kp(KP_L_HIP), img_width)
    upper_arm_score = np.clip(ua_s_b + ua_adj, 1, 6)
    la_s_b, la_adj = score_lower_arm(kp(KP_R_SHOULDER), kp(KP_R_ELBOW), kp(KP_R_WRIST))
    lower_arm_score = np.clip(la_s_b + la_adj, 1, 3)
    w_s_b, w_adj = score_wrist(kp(KP_R_ELBOW), kp(KP_R_WRIST), img_width_for_ref=img_width) # Pass img_width for deviation heuristic
    wrist_score = np.clip(w_s_b + w_adj, 1, 4)
    wrist_twist_score = np.clip(score_wrist_twist(), 1, 2)

    try: posture_score_A = TABLE_A[upper_arm_score-1][lower_arm_score-1][wrist_score-1][wrist_twist_score-1]
    except IndexError: posture_score_A = 7
    score_C_wrist_arm = posture_score_A + arm_muscle_use_score + arm_force_load_score

    nk_s_b, nk_adj = score_neck_v2(kp(KP_NOSE), kp(KP_L_SHOULDER), kp(KP_R_SHOULDER), kp(KP_L_EAR), kp(KP_R_EAR))
    neck_score = np.clip(nk_s_b + nk_adj, 1, 6)
    tk_s_b, tk_adj = score_trunk(kp(KP_L_SHOULDER), kp(KP_R_SHOULDER), kp(KP_L_HIP), kp(KP_R_HIP))
    trunk_score = np.clip(tk_s_b + tk_adj, 1, 6)
    legs_score = score_legs(kp(KP_L_KNEE), kp(KP_R_KNEE), kp(KP_L_ANKLE), kp(KP_R_ANKLE))

    try:
        n_idx, t_idx, l_idx = np.clip(neck_score-1,0,5), np.clip(trunk_score-1,0,len(TABLE_B[0])-1), np.clip(legs_score-1,0,1)
        posture_score_B = TABLE_B[n_idx][t_idx][l_idx]
    except IndexError: posture_score_B = 7
    score_D_ntl = posture_score_B + ntl_muscle_use_score + ntl_force_load_score
    
    score_C_idx = np.clip(score_C_wrist_arm - 1, 0, len(TABLE_C) - 1)
    score_D_idx = np.clip(score_D_ntl - 1, 0, len(TABLE_C[0]) - 1)
    try: final_rula_score = TABLE_C[score_C_idx][score_D_idx]
    except IndexError: final_rula_score = 7

    summary = {
        "UpperArm_Score": upper_arm_score, "LowerArm_Score": lower_arm_score,
        "Wrist_Score": wrist_score, "WristTwist_Score": wrist_twist_score,
        "Posture_Score_A": posture_score_A, "Score_C_WristArm": score_C_wrist_arm,
        "Neck_Score": neck_score, "Trunk_Score": trunk_score, "Legs_Score": legs_score,
        "Posture_Score_B": posture_score_B, "Score_D_NTL": score_D_ntl,
        "FINAL_RULA_SCORE": final_rula_score
    }
    return final_rula_score, summary

# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_filename = sys.argv[1]
        image_filename = "".join(c for c in image_filename if c.isalnum() or c in (' ','_','-')).rstrip()
        image_path = "images/" + image_filename + ".jpg"
        print(f"Attempting to load image from command line: {image_path}")
    else:
        image_path = "images/1.jpg"
        print(f"No command line argument given, using default image: {image_path}")

    try: model = YOLO("yolo11x-pose.pt")
    except Exception as e:
        print(f"Error loading YOLO model: {e}.")
        exit()

    try: results = model(image_path, verbose=False)
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        exit()
    except Exception as e_pred:
        print(f"Error during model prediction on '{image_path}': {e_pred}.")
        exit()

    processed_person = False
    for r_idx, r_val in enumerate(results): # Use enumerate if results can be a list
        if r_val.keypoints and r_val.keypoints.data.numel() > 0:
            person_kpts_xy_normalized = r_val.keypoints.xyn[0].cpu().numpy()
            img_h, img_w = r_val.orig_shape if r_val.orig_shape else (r_val.imgs[0].shape[0],r_val.imgs[0].shape[1])
            person_kpts_pixels = person_kpts_xy_normalized * np.array([img_w, img_h])
            
            # Pass img_width for reference in some heuristics
            final_score, summary = get_rula_score(person_kpts_pixels, img_w,
                                                  arm_muscle_use_score=0, arm_force_load_score=0,
                                                  ntl_muscle_use_score=0, ntl_force_load_score=0)

            if final_score is not None:
                print("\n--- Conceptual RULA Score Summary ---")
                for key, value in summary.items():
                    value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                    print(f"  {key}: {value_str}")
                action_level = ["Acceptable.", "Further investigation, may need changes.", "Investigate and change soon.", "Investigate and change immediately."][np.clip(final_score-1,0,3)//2 if final_score <=7 else 3]
                print(f"  Action Level (Typical): {action_level if final_score <=7 else 'Investigate and change immediately (Score > 7)'}")

            else: print(f"Could not calculate RULA score: {summary}")

            annotated_image = r_val.plot()
            img_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            # img_pil.show()
            processed_person = True
            break 
    
    if not processed_person: print(f"No person with keypoints detected in '{image_path}'.")
    print("\n" + "*"*70 + "\nREMINDER: Conceptual & Simplified RULA. NOT for real assessment.\n" + "*"*70)