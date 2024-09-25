from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# Load the YOLOv8 model
# model = YOLO('yolov10n.pt')  # Ensure you have the model in the correct path
model = YOLO('best_yolov10n.pt')

# Load class names from the 'classes.txt' file
with open('classes.txt', 'r') as f:
    class_names = f.read().splitlines()

# Define the known widths for each object in meters (adjust as needed)
KNOWN_WIDTHS = {
    'Car': 2.0,              # Known width in meters
    'Pedestrian': 0.5,
    'Van': 2.2,
    'Cyclist': 1.5,
    'Truck': 3.0,
    'Tram': 3.5,
    'Person_sitting': 0.6
}

# Color scheme for each class
COLORS = {
    'Car': (255, 0, 0),               # Blue for cars
    'Pedestrian': (0, 255, 0),        # Green for pedestrians
    'Van': (0, 0, 255),               # Red for vans
    'Cyclist': (255, 255, 0),         # Yellow for cyclists
    'Truck': (255, 0, 255),           # Magenta for trucks
    'Tram': (0, 255, 255),            # Cyan for trams
    'Person_sitting': (128, 0, 128)   # Purple for sitting persons
}

FOCAL_LENGTH = 721.5377  # Focal length in pixels for the KITTI dataset

# Function to calculate the distance from the camera
def calculate_distance(focal_length, known_width, pixel_width):
    if pixel_width == 0:
        return None  # Prevent division by zero
    return (known_width * focal_length) / pixel_width

# Function to perform detection and distance calculation
def detect_and_save_image(image_path, output_folder='output_images'):
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image {image_path}")
        return
    
    # Perform object detection
    results = model(image)

    # Parse the results
    detections = results[0].boxes  # Get bounding boxes

    # Iterate through detections and draw bounding boxes for selected classes only
    for det in detections:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = det.conf.item()  # Confidence score, convert tensor to float
        cls = int(det.cls)  # Class index

        if cls < len(class_names):
            class_name = class_names[cls]
            if class_name in class_names:  # Proceed if the class name is in the list

                # Calculate the width of the bounding box (pixel width)
                box_width = x2 - x1

                # Get the known width for this class
                known_width = KNOWN_WIDTHS.get(class_name, 1.0)  # Default width if unknown

                # Calculate the distance using the focal length, known object width, and bounding box width
                distance = calculate_distance(FOCAL_LENGTH, known_width, box_width)

                # Set color for the detected object
                color = COLORS.get(class_name, (255, 255, 255))  # Default white color

                # Label with class name, confidence, and distance
                label = f"{class_name} {conf:.2f}, Distance: {distance:.2f} meters"

                # Draw rectangle and label on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            print(f"Class index {cls} not in 'classes.txt'. Ignored.")

    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Save the image with detections
    output_path = Path(output_folder) / Path(image_path).name
    cv2.imwrite(str(output_path), image)

    print(f"Processed image saved to {output_path}")

# Test with an example image
# detect_and_save_image('test/IMG_20240913_201112.jpg')  # Replace with the actual path to your input image

detect_and_save_image('test/000138.png')  # Replace with the actual path to your input image
