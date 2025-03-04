import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Ensure 'best.pt' is in the same directory

# Open webcam
webcam_video_stream = cv2.VideoCapture(0)

# Create a Matplotlib figure for display
plt.ion()  # Turn on interactive mode for Matplotlib
fig, ax = plt.subplots()

while True:
    ret, current_frame = webcam_video_stream.read()
    if not ret:
        break  # Stop if webcam is not accessible

    # Run YOLOv8 inference on the current frame
    results = model(current_frame)

    # Process results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Draw bounding box
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(current_frame, f"{model.names[cls]} {conf:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0, 255, 0), 2)

    # Convert frame to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    # Clear previous frame and display new one
    ax.clear()
    ax.imshow(frame_rgb)
    ax.axis("off")  # Hide axes
    plt.pause(0.01)  # Pause to refresh the frame

    # Check for exit command
    if plt.waitforbuttonpress(0.01):  # Close on key press
        break

webcam_video_stream.release()
plt.close()  # Close Matplotlib window
