from ultralytics import YOLO
from PIL import Image # Import Pillow library for displaying images
import cv2 # Import OpenCV for displaying images (alternative)

# Load a model
model = YOLO("yolov8s-pose.pt") # Corrected to a standard model name, ensure you have this file or use your "yolo11x-pose.pt"

# Predict with the model
results = model("images/1.jpg")  # predict on an image

# --- Visualize the results ---
for r in results:
    # Set boxes=False to hide bounding boxes
    # Set show_labels=False to hide class labels if they appear
    im_array = r.plot(boxes=False)
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    # To save the image instead of showing:
    # im.save("results_pose_only.jpg")