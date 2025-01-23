import cv2
from picamera2 import Picamera2
import time
import os
# Initialize the camera
picam2 = Picamera2()
# Configure for the highest quality capture
capture_config = picam2.create_still_configuration(
 main={"size": picam2.sensor_resolution, "format": "RGB888"},
 buffer_count=1
)
picam2.configure(capture_config)
# Start the camera
picam2.start()
# Parameters for capturing images
num_images_per_class = 50
classes = ['good', 'defect_hole', 'defect_color']
base_path = '/home/ash/defect_detection/data'
for cls in classes:
 input(f"Press Enter to start capturing images for {cls} class...")
 for i in range(num_images_per_class):
 # Capture the image
 im = picam2.capture_array()

 # Flip the image if needed
 im = cv2.flip(im, -1)

 # Show the image in a window
 cv2.imshow("Camera", im)

 # Save the image with a filename pattern
 cv2.imwrite(f'{base_path}/{cls}/{cls}_{i}.jpg', im)

 # Wait for a short period before capturing the next image
 time.sleep(0.1) # Adjust the delay as needed
 # Break the loop if 'q' is pressed
 if cv2.waitKey(1) == ord('q'):
 break
 print(f"Finished capturing images for {cls} class.")
# Clean up
cv2.destroyAllWindows()
picam2.stop() 
