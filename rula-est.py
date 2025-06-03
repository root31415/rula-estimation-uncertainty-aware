import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import math

# --- Constants for Keypoint Indices (COCO 17 format) ---
# (Based on typical Ultralytics YOLO-Pose output)
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
def get_keypoint(kpts, index):
    """Safely get keypoint, return None if index out of bounds or data invalid."""
    if kpts is not None and index < len(kpts) and kpts[index].sum() != 0: # Check if not (0,0)
        return kpts[index]
    return None

def calculate_angle_2d(p1, p2, p3):
    """Calculates angle p1-p2-p3 in 2D. p2 is the vertex. Returns angle in degrees."""
    if p1 is None or p2 is None or p3 is None: return None
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle_rad = np.arctan2(np.linalg.det([v1,v2]), np.dot(v1,v2)) # Signed angle
    angle_deg = np.degrees(angle_rad)
    # To make it consistent with RULA views (e.g. elbow flexion)
    # This might need adjustment based on expected keypoint configuration
    # For simple flexion/extension like elbow, often want 0-180 range
    # For now, let's ensure it's positive for RULA ranges.
    # A more robust way is to define vectors relative to body segments.
    # For elbow, typically 180 - abs(angle_deg) if using this formula for flexion.
    # Let's use a direct dot product for unsigned angle for flexion.
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    if mag_v1 == 0 or mag_v2 == 0: return None
    angle_cos = np.clip(dot_product / (mag_v1 * mag_v2), -1.0, 1.0)
    return np.degrees(np.arccos(angle_cos))


def get_vertical_angle(p_start, p_end, assumed_person_facing_right=True):
    """Angle of vector p_start -> p_end with the Y-axis (0 deg if pointing down)."""
    if p_start is None or p_end is None: return None
    vector = np.array(p_end) - np.array(p_start)
    if np.linalg.norm(vector) == 0: return None # Avoid division by zero if points are same
    # Vertical vector (pointing down in image coordinates)
    vertical_ref = np.array([0, 1]) # Y positive is downwards
    
    unit_vector = vector / np.linalg.norm(vector)
    dot_product = np.dot(unit_vector, vertical_ref)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    # Determine flexion (+) or extension (-) - this is very view dependent
    # If X component of vector is negative, person leaning "forward" assuming side view facing right
    if assumed_person_facing_right:
        if vector[0] < 0: # Leaning forward relative to the anchor point p_start
            return angle_deg
        else: # Leaning backward
            return -angle_deg # Negative for extension past vertical
    else: # Person facing left
        if vector[0] > 0:
            return angle_deg
        else:
            return -angle_deg
    return angle_deg # Default if no facing direction specified

def get_midpoint(p1, p2):
    if p1 is None or p2 is None: return None
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

# --- RULA Scoring Logic ---

# Step 1 & 2: Upper Arm & Lower Arm
def score_upper_arm(r_shoulder, r_elbow, l_shoulder, r_hip, l_hip, assumed_person_facing_right=True):
    # This is highly simplified. True RULA considers flexion/extension, abduction, and shoulder raising.
    # We'll try to estimate flexion/extension based on angle with an approximate trunk line or vertical.
    # Abduction and shoulder raising are very hard from 2D alone.

    # Approximate trunk line: mid-hip to mid-shoulder
    mid_hip = get_midpoint(l_hip, r_hip)
    mid_shoulder_for_trunk = get_midpoint(l_shoulder, r_shoulder)
    
    if mid_hip is None or mid_shoulder_for_trunk is None or r_shoulder is None or r_elbow is None:
        return 1, 0 # Default to best score, no adjustments

    # Upper arm vector
    vec_upper_arm = np.array(r_elbow) - np.array(r_shoulder)
    if np.linalg.norm(vec_upper_arm) == 0: return 1,0

    # Trunk vector for reference (simplified)
    vec_trunk = np.array(mid_shoulder_for_trunk) - np.array(mid_hip)
    if np.linalg.norm(vec_trunk) == 0: vec_trunk = np.array([0, -1]) # Assume upright if no hip data


    # Angle between trunk and upper arm (flexion/extension approximation)
    # Using dot product to get angle between two vectors
    dot_product = np.dot(vec_upper_arm, vec_trunk)
    mag_ua = np.linalg.norm(vec_upper_arm)
    mag_trunk = np.linalg.norm(vec_trunk)
    
    ua_flex_ext_angle_deg = 0
    if mag_ua > 0 and mag_trunk > 0:
        cos_angle = np.clip(dot_product / (mag_ua * mag_trunk), -1.0, 1.0)
        ua_flex_ext_angle_deg = np.degrees(np.arccos(cos_angle))
        # Sign determination (very heuristic, depends on elbow x relative to shoulder x for side view)
        if assumed_person_facing_right and r_elbow[0] < r_shoulder[0]: # Elbow forward of shoulder
            pass # Positive flexion
        elif not assumed_person_facing_right and r_elbow[0] > r_shoulder[0]:
            pass # Positive flexion
        elif r_elbow[0] == r_shoulder[0] : # Straight down
             ua_flex_ext_angle_deg = 0
        else: # Elbow behind shoulder (extension)
            ua_flex_ext_angle_deg = -ua_flex_ext_angle_deg


    # Score based on RULA flexion/extension ranges
    score = 1
    if 20 < ua_flex_ext_angle_deg <= 45: score = 2
    elif 45 < ua_flex_ext_angle_deg <= 90: score = 3
    elif ua_flex_ext_angle_deg > 90: score = 4
    elif ua_flex_ext_angle_deg < -20: score = 2 # Extension
    elif -20 <= ua_flex_ext_angle_deg < 0 : score = 2 # Slight Extension, RULA often groups with 0-20 flexion but can be +1 adjustment for >20 ext. Simplified here.

    # Adjustments (Simplified: shoulder raised or arm abducted - assume +0 for now)
    # A true abduction check would require frontal plane view or more robust 3D.
    # Shoulder raised: if r_shoulder Y is much higher than l_shoulder Y (or vice versa) if not symmetrical task
    adj_shoulder_raised = 0
    adj_abducted = 0
    # adj_supported_or_leaning = 0 # -1 if supported

    # Example very naive abduction check (angle of upper arm with vertical when trunk is vertical)
    # This is not true abduction if trunk is also tilted.
    # For now, returning 0 for these complex adjustments.
    
    return score, (adj_shoulder_raised + adj_abducted) # adj_supported_or_leaning would be negative

def score_lower_arm(r_shoulder, r_elbow, r_wrist):
    elbow_angle = calculate_angle_2d(r_shoulder, r_elbow, r_wrist)
    if elbow_angle is None: return 1, 0 # Default

    # RULA: Score 1 if 60-100 deg. Score 2 if <60 or >100.
    # Note: RULA uses angle of flexion, so 180 - elbow_angle if elbow_angle is inner angle.
    # Our calculate_angle_2d gives the angle at the joint, so if it's 90, that's good.
    # RULA's "60-100 degrees" means the arm is bent, not straight.
    # An elbow_angle from calculate_angle_2d of 0-20 would be very straight (extended).
    # An elbow_angle of 160-180 would be very flexed.
    # RULA's "lower arm position" is often visualized as angle from upper arm line.
    # Let's assume our elbow_angle is the inner angle. RULA expects "working position"
    # Let's interpret RULA's 60-100 degrees as the deviation from the straight line of the upper arm.
    # If our elbow_angle from calculate_angle_2d is the actual angle between bones:
    #   0 degrees = fully extended; 180 degrees = fully flexed touching shoulder
    # RULA's optimal 60-100 is for the "working angle" often depicted.
    # If our angle is 0-20 deg (straight), RULA score 2. If 160-180 deg (fully flexed), RULA score 2.
    # If 80-120 deg (using common interpretation of keypoint angle for elbow), this is RULA 1.
    # Let's assume our `calculate_angle_2d` for elbow gives about 90 for a right angle.
    # RULA scores: 1 for 60-100 deg flexion. 2 for <60 or >100.
    # This needs to be clear: if our angle is the "inner angle" at elbow:
    #   angle ~0-20 (arm straight) -> RULA score 2
    #   angle ~160-180 (arm fully bent) -> RULA score 2
    #   angle ~70-120 (mid-range) -> RULA score 1
    # This matches RULA's intent.
    
    score = 1
    if 60 <= elbow_angle <= 100: # Optimal range
        score = 1
    else: # Outside optimal range
        score = 2
    
    # Adjustment: Arm working across midline or out to side (simplified to 0)
    adj_midline_or_side = 0
    return score, adj_midline_or_side

# Step 3 & 4: Wrist & Wrist Twist
def score_wrist(r_elbow, r_wrist, r_hand_knuckle_approx):
    adj_deviation = 0 # Default deviation adjustment
    if r_hand_knuckle_approx is None:
        # MODIFICATION: If no hand point, assume a less optimal wrist posture
        # Instead of best-case score 1, let's assign a base score of 2 or 3.
        # This reflects that a perfectly neutral wrist is less likely if not explicitly measured.
        # Let's try defaulting to a base score of 2 for flexion/extension.
        score = 2 # Assume mild flexion/extension due to lack of data
        # adj_deviation could also be set to +1 if side view is unclear for deviation
        # For now, just making flexion/extension less optimal by default.
    else:
        wrist_angle_calc = calculate_angle_2d(r_elbow, r_wrist, r_hand_knuckle_approx)
        # Adjust angle to be relative to forearm's line (0 = straight wrist)
        wrist_angle_deg = abs(180 - wrist_angle_calc) if wrist_angle_calc is not None else 0

        if wrist_angle_deg <= 5 : score = 1 # RULA: 0 degrees = Score 1
        elif wrist_angle_deg <= 15: score = 2 # RULA: +/- 15 degrees = Score 2
        else: score = 3 # RULA: > 15 degrees = Score 3
        # Score 4 in RULA for wrist is if >15 AND bent/twisted. We handle twist separately.

    # adj_deviation: Ulnar/Radial Deviation. Still hard. Simplified to 0.
    # If you could estimate it:
    # if deviation_angle > 15 degrees: adj_deviation = 1
    
    return score, adj_deviation

def score_wrist_twist():
    # Extremely hard from 2D. Assume Score 1 (mid-range twist) or 2 if markedly twisted.
    # For this conceptual code, always assume 1 (neutral/best).
    return 1

# Step 9: Neck Score
def score_neck(kp_nose, kp_l_shoulder, kp_r_shoulder, kp_l_ear, kp_r_ear, assumed_person_facing_right=True):
    # Neck flexion/extension
    # Use midpoint of shoulders as base of neck, and midpoint of ears/nose as head position
    mid_shoulder = get_midpoint(kp_l_shoulder, kp_r_shoulder)
    head_pt = kp_nose # or get_midpoint(kp_l_ear, kp_r_ear)
    if head_pt is None: head_pt = get_midpoint(kp_l_ear, kp_r_ear)

    if mid_shoulder is None or head_pt is None: return 1, 0 # Default

    neck_angle_deg = get_vertical_angle(mid_shoulder, head_pt, assumed_person_facing_right)
    if neck_angle_deg is None: neck_angle_deg = 0

    score = 1
    if 0 <= neck_angle_deg <= 10: score = 1
    elif 10 < neck_angle_deg <= 20: score = 2
    elif neck_angle_deg > 20: score = 3
    elif neck_angle_deg < 0: score = 4 # Extension

    # Adjustments: Neck twisted or side-bending (Simplified to 0)
    adj_twist_side_bend = 0
    return score, adj_twist_side_bend

# Step 10: Trunk Score
def score_trunk(kp_l_shoulder, kp_r_shoulder, kp_l_hip, kp_r_hip, assumed_person_facing_right=True):
    mid_shoulder = get_midpoint(kp_l_shoulder, kp_r_shoulder)
    mid_hip = get_midpoint(kp_l_hip, kp_r_hip)

    if mid_shoulder is None or mid_hip is None: return 1, 0 # Default

    trunk_angle_deg = get_vertical_angle(mid_hip, mid_shoulder, assumed_person_facing_right)
    if trunk_angle_deg is None: trunk_angle_deg = 0
    
    score = 1
    # RULA: 0 deg (upright) = 1. 0-20 deg flexion = 2. 20-60 deg = 3. >60 deg = 4. Extension = +1 to score.
    if 0 <= trunk_angle_deg <= 10: score = 1 # Slightly more lenient for "upright"
    elif 10 < trunk_angle_deg <= 20: score = 2
    elif 20 < trunk_angle_deg <= 60: score = 3
    elif trunk_angle_deg > 60: score = 4
    
    if trunk_angle_deg < 0: # Extension
        if score <=2 : score = 2 # If upright or slightly flexed but also extension (edge case), effectively score 2
        else: score = score +1 # Typically add 1 for extension, here maps to next risk band


    # Adjustments: Trunk twisted or side-bending (Simplified to 0)
    adj_twist_side_bend = 0
    return score, adj_twist_side_bend

# Step 11: Legs Score
def score_legs(kp_l_knee, kp_r_knee, kp_l_ankle, kp_r_ankle):
    # RULA: 1 if legs and feet well supported and symmetrical. 2 if not.
    # This is very hard to judge from keypoints alone without context.
    # Simplified: check if both knees and ankles are detected.
    if (get_keypoint([kp_l_knee, kp_r_knee, kp_l_ankle, kp_r_ankle],0) is not None and
        get_keypoint([kp_l_knee, kp_r_knee, kp_l_ankle, kp_r_ankle],1) is not None and
        get_keypoint([kp_l_knee, kp_r_knee, kp_l_ankle, kp_r_ankle],2) is not None and
        get_keypoint([kp_l_knee, kp_r_knee, kp_l_ankle, kp_r_ankle],3) is not None):
        return 1 # Assume supported if all keypoints present (very naive)
    return 2


# --- RULA Tables ---
# Table A: Wrist and Arm Analysis
# Rows: Upper Arm Score (1-6)
# Columns: Lower Arm Score (1-3)
# Cells: Wrist Score (1-4) -> Wrist Twist Score (1-2) -> Posture Score A
# This is a 4D table. We'll use nested dicts or direct indexing.
# Format: table_A[ua_score-1][la_score-1][wrist_score-1][wrist_twist_score-1]
TABLE_A = [
    [[[1,2],[2,2]],[[2,2],[2,3]],[[2,3],[3,3]],[[3,3],[3,4]]], # UA 1
    [[[2,2],[2,3]],[[3,3],[3,3]],[[3,3],[3,4]],[[4,4],[4,4]]], # UA 2
    [[[3,3],[3,3]],[[3,3],[3,4]],[[3,4],[4,4]],[[4,4],[4,5]]], # UA 3
    [[[3,4],[4,4]],[[4,4],[4,4]],[[4,4],[4,5]],[[5,5],[5,5]]], # UA 4
    [[[4,4],[4,4]],[[4,4],[4,5]],[[4,5],[5,5]],[[5,5],[5,6]]], # UA 5
    [[[4,4],[4,5]],[[4,5],[5,5]],[[5,5],[5,6]],[[5,6],[6,7]]]  # UA 6
]

# Table B: Neck, Trunk and Leg Analysis
# Rows: Neck Score (1-6)
# Columns: Trunk Score (1-6)
# Cells: Leg Score (1-2) -> Posture Score B
# Format: table_B[neck_score-1][trunk_score-1][leg_score-1]
TABLE_B = [
    [[1,3],[2,3],[3,4],[4,5],[5,5],[6,6],[7,7]], # Neck 1 (Trunk 1-7, Leg 1-2)
    [[2,3],[2,3],[3,4],[4,5],[5,5],[6,7],[7,7]], # Neck 2
    [[3,3],[3,4],[4,4],[4,5],[5,6],[6,7],[7,7]], # Neck 3
    [[3,4],[4,4],[4,5],[5,5],[6,6],[7,7],[7,7]], # Neck 4
    [[4,4],[4,5],[5,5],[5,6],[6,7],[7,7],[7,7]], # Neck 5
    [[4,5],[5,5],[5,6],[6,6],[6,7],[7,7],[7,7]]  # Neck 6
]


# Table C: Final RULA Score
# Rows: Wrist & Arm Score (Score C from Table A + muscle/force) (1-8+)
# Columns: Neck, Trunk & Leg Score (Score D from Table B + muscle/force) (1-8+)
# Cells: Grand RULA Score (1-7)
# Format: table_C[score_C_idx][score_D_idx]
TABLE_C = [
    [1,2,3,3,4,5,5], # Score C = 1
    [2,2,3,4,4,5,5], # Score C = 2
    [3,3,3,4,4,5,6], # Score C = 3
    [3,3,4,4,5,6,6], # Score C = 4
    [4,4,4,5,6,6,7], # Score C = 5
    [4,4,5,5,6,7,7], # Score C = 6
    [5,5,6,6,7,7,7], # Score C = 7
    [5,5,6,7,7,7,7]  # Score C = 8+
]

def get_rula_score(kpts,
                   arm_muscle_use_score=0, arm_force_load_score=0, # For Table A result
                   ntl_muscle_use_score=0, ntl_force_load_score=0): # For Table B result
    """
    Calculates a conceptual RULA score based on keypoints and optional muscle/force scores.
    REMEMBER: This is highly simplified and not for real ergonomic assessment.
    Keypoints are assumed to be for a single person.
    Muscle use scores: 0 (intermittent) or 1 (static/repetitive).
    Force/Load scores: 0 (<4.4lb intermittent), 1 (4.4-22lb intermittent OR <4.4lb static/rep),
                       2 (4.4-22lb static/rep OR >22lb intermittent), 3 (>22lb static/rep).
    """
    if kpts is None or len(kpts) < 17:
        return None, "Not enough keypoints."

    # Extract keypoints for clarity (assuming right side for arm analysis)
    # A full system might analyze both and take worst, or if specific side is of interest.
    l_shoulder = get_keypoint(kpts, KP_L_SHOULDER)
    r_shoulder = get_keypoint(kpts, KP_R_SHOULDER)
    l_elbow = get_keypoint(kpts, KP_L_ELBOW)
    r_elbow = get_keypoint(kpts, KP_R_ELBOW)
    l_wrist = get_keypoint(kpts, KP_L_WRIST)
    r_wrist = get_keypoint(kpts, KP_R_WRIST)
    # For wrist angle, we need a point representing the hand/knuckles.
    # This is a major simplification: assume hand is aligned with wrist if no knuckle point.
    # Or, one could try to estimate it if wrist and elbow are far apart, pointing "out".
    # For this conceptual code, we'll pass None, forcing score_wrist to assume neutral.
    r_hand_knuckle_approx = None # Placeholder

    l_hip = get_keypoint(kpts, KP_L_HIP)
    r_hip = get_keypoint(kpts, KP_R_HIP)
    nose = get_keypoint(kpts, KP_NOSE)
    l_ear = get_keypoint(kpts, KP_L_EAR)
    r_ear = get_keypoint(kpts, KP_R_EAR)
    l_knee = get_keypoint(kpts, KP_L_KNEE)
    r_knee = get_keypoint(kpts, KP_R_KNEE)
    l_ankle = get_keypoint(kpts, KP_L_ANKLE)
    r_ankle = get_keypoint(kpts, KP_R_ANKLE)

    # --- Section A: Arm & Wrist Analysis (Using Right Arm as example) ---
    ua_score_base, ua_adj = score_upper_arm(r_shoulder, r_elbow, l_shoulder, r_hip, l_hip)
    upper_arm_score = ua_score_base + ua_adj
    upper_arm_score = np.clip(upper_arm_score, 1, 6) # RULA Upper Arm score is 1-6

    la_score_base, la_adj = score_lower_arm(r_shoulder, r_elbow, r_wrist)
    lower_arm_score = la_score_base + la_adj
    lower_arm_score = np.clip(lower_arm_score, 1, 3) # RULA Lower Arm score is 1-3

    w_score_base, w_adj = score_wrist(r_elbow, r_wrist, r_hand_knuckle_approx)
    wrist_score = w_score_base + w_adj
    wrist_score = np.clip(wrist_score, 1, 4) # RULA Wrist score is 1-4

    wrist_twist_score = score_wrist_twist() # Simplified to 1 or 2
    wrist_twist_score = np.clip(wrist_twist_score, 1, 2)

    # Lookup in Table A
    try:
        posture_score_A = TABLE_A[upper_arm_score-1][lower_arm_score-1][wrist_score-1][wrist_twist_score-1]
    except IndexError:
        print("Error looking up Table A. Check input scores or table definition.")
        posture_score_A = 7 # Default to a high score on error
        
    score_C_wrist_arm = posture_score_A + arm_muscle_use_score + arm_force_load_score

    # --- Section B: Neck, Trunk & Leg Analysis ---
    neck_score_base, neck_adj = score_neck(nose, l_shoulder, r_shoulder, l_ear, r_ear)
    neck_score = neck_score_base + neck_adj
    neck_score = np.clip(neck_score, 1, 6) # RULA Neck score 1-6

    trunk_score_base, trunk_adj = score_trunk(l_shoulder, r_shoulder, l_hip, r_hip)
    trunk_score = trunk_score_base + trunk_adj
    trunk_score = np.clip(trunk_score, 1, 6) # RULA Trunk score 1-6

    legs_score = score_legs(l_knee, r_knee, l_ankle, r_ankle) # 1 or 2

    # Lookup in Table B
    try:
        # Ensure scores are within valid index ranges for TABLE_B
        # Neck score can be up to 6. Trunk up to 6. Leg 1 or 2.
        # Table B: neck_score (1-6), trunk_score (1-6), leg_score (1-2)
        # Max indices: neck=5, trunk=5 (for score 6), leg=1 (for score 2)
        # The RULA sheet sometimes shows Trunk up to 7 for extreme postures (includes +1 for twist/sidebend)
        # Our TABLE_B definition has 7 columns for trunk, which is good.
        
        # Clip scores to ensure they are valid indices for the table
        n_idx = np.clip(neck_score - 1, 0, len(TABLE_B) - 1)
        t_idx = np.clip(trunk_score - 1, 0, len(TABLE_B[0]) - 1) # Max trunk score index
        l_idx = np.clip(legs_score - 1, 0, len(TABLE_B[0][0]) - 1) # Max leg score index
        
        posture_score_B = TABLE_B[n_idx][t_idx][l_idx]

    except IndexError:
        print(f"Error looking up Table B. Neck:{neck_score}, Trunk:{trunk_score}, Legs:{legs_score}. Check inputs/table.")
        posture_score_B = 7 # Default high

    score_D_ntl = posture_score_B + ntl_muscle_use_score + ntl_force_load_score
    
    # --- Table C: Final RULA Score ---
    # Scores C and D can range from 1 up to 8+ (RULA sheet typically shows 1-7, then 8+)
    # We'll clip to max index of Table C
    score_C_idx = np.clip(score_C_wrist_arm - 1, 0, len(TABLE_C) - 1)
    score_D_idx = np.clip(score_D_ntl - 1, 0, len(TABLE_C[0]) - 1)

    try:
        final_rula_score = TABLE_C[score_C_idx][score_D_idx]
    except IndexError:
        print("Error looking up Table C. Check input scores C/D or table definition.")
        final_rula_score = 7 # Default high

    # Store intermediate scores for understanding
    analysis_summary = {
        "UpperArm_Score_Base": ua_score_base, "UpperArm_Adj": ua_adj, "UpperArm_Total": upper_arm_score,
        "LowerArm_Score_Base": la_score_base, "LowerArm_Adj": la_adj, "LowerArm_Total": lower_arm_score,
        "Wrist_Score_Base": w_score_base, "Wrist_Adj": w_adj, "Wrist_Total": wrist_score,
        "WristTwist_Score": wrist_twist_score,
        "Posture_Score_A": posture_score_A,
        "Arm_MuscleUse_Score": arm_muscle_use_score, "Arm_ForceLoad_Score": arm_force_load_score,
        "Score_C_WristArm": score_C_wrist_arm,
        "Neck_Score_Base": neck_score_base, "Neck_Adj": neck_adj, "Neck_Total": neck_score,
        "Trunk_Score_Base": trunk_score_base, "Trunk_Adj": trunk_adj, "Trunk_Total": trunk_score,
        "Legs_Score": legs_score,
        "Posture_Score_B": posture_score_B,
        "NTL_MuscleUse_Score": ntl_muscle_use_score, "NTL_ForceLoad_Score": ntl_force_load_score,
        "Score_D_NTL": score_D_ntl,
        "FINAL_RULA_SCORE": final_rula_score
    }

    return final_rula_score, analysis_summary


# --- Main Execution ---
if __name__ == "__main__":
    # Load YOLO model
    try:
        model = YOLO("yolo11x-pose.pt") # Or your preferred YOLO-Pose model
    except Exception as e:
        print(f"Error loading YOLO model: {e}. Ensure it's installed or path is correct.")
        exit()

    image_path = "8.jpg" # Replace with your image
    
    # Attempt to load image and run prediction
    try:
        results = model(image_path)
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        # Fallback to download a sample image if one isn't found
        try:
            import requests
            print("Attempting to download a sample 'bus.jpg' as 'pose-test.jpg'...")
            url = "https://ultralytics.com/images/bus.jpg" 
            response = requests.get(url)
            if response.status_code == 200:
                with open("pose-test.jpg", "wb") as f: f.write(response.content)
                print("Sample image downloaded. Please replace with a relevant human pose image for RULA.")
                results = model("pose-test.jpg")
            else:
                print("Failed to download sample image. Exiting.")
                exit()
        except Exception as e_download:
            print(f"Could not download sample image: {e_download}. Exiting.")
            exit()
    except Exception as e_pred:
        print(f"Error during model prediction: {e_pred}. Exiting.")
        exit()


    # Process first detected person with keypoints
    processed_person = False
    for r in results:
        if r.keypoints and r.keypoints.data.numel() > 0:
            # Get keypoints for the first person detected
            # kpts_data.xy is a list of tensors, one for each person.
            # kpts_data.xy[0] is the tensor for the first person.
            person_kpts_xy_normalized = r.keypoints.xyn[0].cpu().numpy() # Normalized [0,1]
            
            # Denormalize if needed (using original image dimensions)
            # If your angle functions expect pixel coords, you need to denormalize.
            # Our current angle functions work with any coordinate system as long as it's consistent.
            # However, for visualization, pixel coords are better.
            # Let's assume the functions can handle normalized or pixel, but for display, denormalize.
            
            img_h, img_w = r.orig_shape if r.orig_shape else (720,1280) # Get original image shape
            person_kpts_pixels = person_kpts_xy_normalized * np.array([img_w, img_h])

            print("\nProcessing RULA for the first detected person...")
            
            # --- YOU MUST PROVIDE THESE SCORES BASED ON ACTUAL TASK ASSESSMENT ---
            # Defaulting to 0 (minimal risk) for this conceptual example.
            # Muscle Use: 0 (intermittent), 1 (static/repetitive)
            # Force/Load: 0 (<2kg int.), 1 (2-10kg int. OR <2kg static),
            #             2 (2-10kg static OR >10kg int.), 3 (>10kg static)
            custom_arm_muscle_use = 0
            custom_arm_force_load = 0
            custom_ntl_muscle_use = 0 # Neck/Trunk/Leg muscle use
            custom_ntl_force_load = 0 # Neck/Trunk/Leg force/load
            
            # Pass pixel coordinates to RULA scoring
            final_score, summary = get_rula_score(person_kpts_pixels,
                                                  arm_muscle_use_score=custom_arm_muscle_use,
                                                  arm_force_load_score=custom_arm_force_load,
                                                  ntl_muscle_use_score=custom_ntl_muscle_use,
                                                  ntl_force_load_score=custom_ntl_force_load)

            if final_score is not None:
                print("\n--- Conceptual RULA Score Summary ---")
                for key, value in summary.items():
                    if isinstance(value, float): value_str = f"{value:.2f}"
                    else: value_str = str(value)
                    print(f"  {key}: {value_str}")
                
                print(f"\n  >>>> OVERALL CONCEPTUAL RULA GRAND SCORE: {final_score} <<<<")

                action_level = ""
                if final_score <= 2: action_level = "Acceptable posture."
                elif final_score <= 4: action_level = "Further investigation needed, may need changes."
                elif final_score <= 6: action_level = "Investigate and implement changes soon."
                else: action_level = "Investigate and implement changes immediately."
                print(f"  Action Level (Typical): {action_level}")
                
            else:
                print(f"Could not calculate RULA score: {summary}") # Summary here would be error message

            # Display the image with plotted keypoints
            annotated_image = r.plot() # Returns a BGR numpy array
            img_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            img_pil.show()
            # img_pil.save("conceptual_rula_output.jpg")

            processed_person = True
            break # Process only the first person for this example
    
    if not processed_person:
        print("No person with sufficient keypoints detected in the image.")

    print("\n" + "*"*80)
    print("IMPORTANT REMINDER:")
    print("This code is a conceptual illustration with significant simplifications.")
    print("It is NOT a validated ergonomic tool and should NOT be used for safety or health decisions.")
    print("Real RULA assessment requires 3D analysis, expert judgment, and consideration of all ergonomic factors.")
    print("*"*80)