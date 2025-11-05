import tkinter as tk
from PIL import Image, ImageTk
import pycozmo

def stream_camera(cli):
    # Create the main window
    root = tk.Tk()
    root.title("Cozmo Camera Feed")
    
    # Create a label to display the image
    label = tk.Label(root)
    label.pack()

    # Last image received from the robot
    last_im = None

    def on_camera_image(cli, new_im):
        """ Handle new images, coming from the robot. """
        nonlocal last_im
        last_im = new_im

    def update_image():
        """ Update the displayed image """
        nonlocal last_im
        if last_im:
            # Get last image (already 320x240)
            im = last_im
            # Mirror the image
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
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

with pycozmo.connect(enable_procedural_face=False) as cli:
    stream_camera(cli)