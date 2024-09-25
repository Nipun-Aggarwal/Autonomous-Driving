from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('best_yolov10n.pt')
print("Model Loaded")

# Open the video file
video_path = "test/IMG_2754.MOV"
video_capture = cv2.VideoCapture(video_path)

# Get the original video frame rate and dimensions
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the VideoWriter to save the output
output_path = "output_video.MOV"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the codec
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Set the frame skipping factor (e.g., skip every 3 frames)
frame_skip = 3
frame_count = 0

# Process the video frame by frame
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Only process every nth frame (skip others)
    if frame_count % frame_skip == 0:
        # Run YOLO detection on the current frame
        results = model(frame)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Save the annotated frame to the output video
        output_video.write(annotated_frame)

    frame_count += 1

# Release the video capture and writer objects
video_capture.release()
output_video.release()

print("Result saved to", output_path)
