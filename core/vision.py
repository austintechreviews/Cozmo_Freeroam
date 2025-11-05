import tkinter as tk
from PIL import Image, ImageTk
import pycozmo
import cv2
import numpy as np
import os
from datetime import datetime

# Create directories for saved images if they don't exist
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'captured_images')
TRAINING_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'training_images')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize SIFT detector for cube symbol recognition
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

class CubeRecognizer:
    def __init__(self):
        self.cube_templates = {}
        self.load_cube_templates()

    def load_cube_templates(self):
        """Load the template images for cube recognition"""
        template_dir = os.path.join(TRAINING_DIR, 'cube_templates')
        os.makedirs(template_dir, exist_ok=True)
        
        # Load existing templates
        for file in os.listdir(template_dir):
            if file.endswith('.jpg'):
                cube_id = file.split('.')[0]
                template = cv2.imread(os.path.join(template_dir, file), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.cube_templates[cube_id] = {
                        'image': template,
                        'keypoints': None,
                        'descriptors': None
                    }
                    # Compute SIFT features for the template
                    self.cube_templates[cube_id]['keypoints'], self.cube_templates[cube_id]['descriptors'] = \
                        sift.detectAndCompute(template, None)

    def save_cube_template(self, image, cube_id):
        """Save a new cube template"""
        template_path = os.path.join(TRAINING_DIR, 'cube_templates', f'{cube_id}.jpg')
        cv2.imwrite(template_path, image)
        self.load_cube_templates()  # Reload templates

    def recognize_cube(self, image):
        """Try to recognize a cube in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray, None)
        
        best_match = None
        max_good_matches = 0
        
        for cube_id, template in self.cube_templates.items():
            if template['descriptors'] is not None and des2 is not None:
                matches = bf.knnMatch(template['descriptors'], des2, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                        
                if len(good_matches) > max_good_matches:
                    max_good_matches = len(good_matches)
                    best_match = cube_id
        
        if max_good_matches >= 10:  # Minimum threshold for a match
            return best_match
        return None

def stream_camera(cli):
    # Create the main window
    root = tk.Tk()
    root.title("Cozmo Camera Feed")
    
    # Create a label to display the image
    label = tk.Label(root)
    label.pack()

    # Create buttons frame
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Last image received from the robot
    last_im = None
    
    # Initialize cube recognizer
    cube_recognizer = CubeRecognizer()

    def save_image():
        """Save the current frame with detected objects"""
        nonlocal last_im
        if last_im:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_DIR, f'capture_{timestamp}.jpg')
            last_im.save(save_path)
            print(f"Image saved to {save_path}")

    def save_cube_template():
        """Save current frame as a cube template"""
        nonlocal last_im
        if last_im:
            # Convert PIL image to OpenCV format
            opencv_img = cv2.cvtColor(np.array(last_im), cv2.COLOR_RGB2BGR)
            cube_id = f"cube_{len(cube_recognizer.cube_templates) + 1}"
            cube_recognizer.save_cube_template(opencv_img, cube_id)
            print(f"Saved new cube template: {cube_id}")

    # Create buttons
    save_button = tk.Button(button_frame, text="Save Image", command=save_image)
    save_button.pack(side=tk.LEFT, padx=5)
    
    template_button = tk.Button(button_frame, text="Save as Cube Template", command=save_cube_template)
    template_button.pack(side=tk.LEFT, padx=5)
    
    def adjust_brightness(image, brightness_factor=1.5):
        """Adjust the brightness of an image"""
        # Convert to HSV for better brightness control
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Scale the V channel
        hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_factor, 0, 255).astype(np.uint8)
        # Convert back to BGR
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def preprocess_image(image):
        """Enhance image for better face detection"""
        # Apply histogram equalization to improve contrast
        image = cv2.equalizeHist(image)
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        return image

    def on_camera_image(cli, new_im):
        """ Handle new images, coming from the robot. """
        nonlocal last_im
        last_im = new_im

    def adjust_brightness(image, brightness_factor=2):
        """Adjust the brightness of an image"""
        # Convert to HSV for better brightness control
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Scale the V channel
        hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_factor, 0, 255).astype(np.uint8)
        # Convert back to BGR
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def preprocess_image(image):
        """Enhance image for better face detection"""
        # Apply histogram equalization to improve contrast
        image = cv2.equalizeHist(image)
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        return image

    def detect_objects(image):
        """Perform face detection and cube recognition on the image"""
        # Convert PIL image to OpenCV format
        opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Increase brightness
        opencv_img = adjust_brightness(opencv_img)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        
        # Enhance image
        gray = preprocess_image(gray)
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More gradual scaling
            minNeighbors=6,    # More strict detection
            minSize=(50, 50),  # Minimum face size
            maxSize=(200, 200) # Maximum face size
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(opencv_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(opencv_img, 'Face', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Try to recognize cubes
        cube_id = cube_recognizer.recognize_cube(opencv_img)
        if cube_id:
            # Add cube label to the image
            cv2.putText(opencv_img, f"Cube: {cube_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return opencv_img



    # Create buttons
    save_button = tk.Button(button_frame, text="Save Image", command=save_image)
    save_button.pack(side=tk.LEFT, padx=5)
    
    template_button = tk.Button(button_frame, text="Save as Cube Template", command=save_cube_template)
    template_button.pack(side=tk.LEFT, padx=5)

    def update_image():
        """ Update the displayed image """
        nonlocal last_im
        if last_im:
            # Get last image (already 320x240)
            im = last_im
            # Mirror the image
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Process image (face detection and cube recognition)
            opencv_img = detect_objects(im)
            
            # Convert back to PIL for display
            im = Image.fromarray(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
            
            # Convert to PhotoImage for tkinter
            photo = ImageTk.PhotoImage(image=im)
            # Update the label
            label.configure(image=photo)
            label.image = photo  # Keep a reference!
        
        # Schedule the next update
        root.after(50, update_image)  # Update every 50ms (approx 20 FPS)

    try:
        with pycozmo.connect(enable_procedural_face=False) as cli:
            # Raise head
            angle = (pycozmo.robot.MAX_HEAD_ANGLE.radians - pycozmo.robot.MIN_HEAD_ANGLE.radians) / 2.0
            cli.set_head_angle(angle)

            # Register to receive new camera images
            cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

            # Enable camera
            cli.enable_camera()

            # Start the image update loop
            update_image()

            # Start the tkinter main loop
            root.mainloop()

    except KeyboardInterrupt:
        print("\nStream stopped by user")
        root.quit()
    except Exception as e:
        print(f"Error: {e}")
        root.quit()

if __name__ == "__main__":
    stream_camera(None)