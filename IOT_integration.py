import argparse
import sys
import time
import threading
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize
from picamera2 import Picamera2
from adafruit_servokit import ServoKit
# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
frame_lock = threading.Lock()
latest_frame = None
# Initialize the PCA9685 module using the default address (0x40)
kit = ServoKit(channels=16) 
# Define the servo channels
servo_channels = [0, 1, 2, 3, 8, 9]
# Define the MG996R servo properties
servo_min_angle = 0 # Minimum angle for MG996R
servo_max_angle = 180 # Maximum angle for MG996R
# Define positions
positions = {
 'home': [60, 53, 45, 60, 30, 90],
 'pick': [47, 52, 70, 50, 20, 80],
 'good': [56, 70, 40, 90, 30, 90],
 'defect_color': [67, 80, 90, 10, 40, 90],
 'defect_hole': [40, 70, 30, 30, 60, 90],
}
# Function to set servo angle
def set_servo_angle(channel, angle):
 if servo_min_angle <= angle <= servo_max_angle:
 kit.servo[channel].angle = angle
 print(f"Servo {channel} set to {angle} degrees.")
 else:
 print(f"Angle {angle} out of range. Must be between {servo_min_angle} and {servo_max_angle}
degrees.")
# Function to move to a predefined position
def move_to_position(position_name):
 if position_name in positions:
 for i, angle in enumerate(positions[position_name]):
 set_servo_angle(servo_channels[i], angle)
 else:
 print(f"Position {position_name} not defined.")
# Function to capture frames
def capture_frames():
 global latest_frame
 while True:
 frame = picam2.capture_array()
 with frame_lock:
 latest_frame = frame
# Function to handle detection results
def handle_detection(result):
 class_name = result.classifications[0].classes[0].class_name
 print(f"Detected: {class_name}")
 if class_name == 'good':
 move_to_position('good')
 elif class_name == 'defect_color':
 move_to_position('defect_color')
 elif class_name == 'defect_hole':
 move_to_position('defect_hole')
 else:
 print("Unknown class")
# Function to save result
def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms:
int):
 global FPS, COUNTER, START_TIME
# Calculate the FPS
 if COUNTER % fps_avg_frame_count == 0:
 FPS = fps_avg_frame_count / (time.time() - START_TIME)
 START_TIME = time.time()
 detection_result_list.append(result)
 COUNTER += 1
 handle_detection(result.detections[0]) # Assume the first detection is the relevant one
# Main function to run the object detection
def run(model: str, max_results: int, score_threshold: float, camera_id: int, width: int, height: int) ->
None:
 """Continuously run inference on images acquired from the camera."""
 # Visualization parameters
 row_size = 50 # pixels
 left_margin = 24 # pixels
 text_color = (0, 0, 0) # black
 font_size = 1
 font_thickness = 1
 fps_avg_frame_count = 10
 detection_frame = None
 detection_result_list = []
 # Initialize the object detection model
 base_options = python.BaseOptions(model_asset_path=model)
 options = vision.ObjectDetectorOptions(base_options=base_options,
 running_mode=vision.RunningMode.LIVE_STREAM,
 max_results=max_results, score_threshold=score_threshold,
 result_callback=save_result)
 detector = vision.ObjectDetector.create_from_options(options)
 # Start the frame capture thread
 frame_thread = threading.Thread(target=capture_frames, daemon=True)
 frame_thread.start()
 # Continuously capture images from the camera and run inference
 while True:
 with frame_lock:
 if latest_frame is None:
 continue
 image = latest_frame.copy()
 image = cv2.resize(image, (width, height))
 image = cv2.flip(image, -1)
 # Convert the image from BGR to RGB as required by the TFLite model.
 rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
 # Run object detection using the model.
 detector.detect_async(mp_image, time.time_ns() // 1_000_000)
 # Show the FPS
 fps_text = 'FPS = {:.1f}'.format(FPS)
 text_location = (left_margin, row_size) 
current_frame = image
 cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
 font_size, text_color, font_thickness, cv2.LINE_AA)
 if detection_result_list:
 current_frame = visualize(current_frame, detection_result_list[0])
 detection_frame = current_frame
 detection_result_list.clear()
 if detection_frame is not None:
 cv2.imshow('object_detection', detection_frame)
 # Stop the program if the ESC key is pressed.
 if cv2.waitKey(1) == 27:
 break
 detector.close()
 cv2.destroyAllWindows()
# Argument parser for command line arguments
def main():
 parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 parser.add_argument('--model', help='Path of the object detection model.', required=False,
default='best.tflite')
 parser.add_argument('--maxResults', help='Max number of detection results.', required=False,
default=5)
 parser.add_argument('--scoreThreshold', help='The score threshold of detection results.',
required=False, type=float, default=0.25)
 parser.add_argument('--cameraId', help='Id of camera.', required=False, type=int, default=0)
 parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False,
type=int, default=640)
 parser.add_argument('--frameHeight', help='Height of frame to capture from camera.',
required=False, type=int, default=480)
 args = parser.parse_args()
 run(args.model, int(args.maxResults), args.scoreThreshold, int(args.cameraId), args.frameWidth,
args.frameHeight)
if __name__ == '__main__':
 main() 
