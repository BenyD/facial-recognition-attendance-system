import cv2
import time
from utils import create_directories, save_face_data, detect_faces

def list_available_cameras():
    """List all available camera devices"""
    available_cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        ret, _ = cap.read()
        if ret:
            available_cameras.append(index)
        cap.release()
        index += 1
    return available_cameras

def register_face(camera_index=0):
    # Create necessary directories
    create_directories()
    
    # List available cameras
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("No cameras found!")
        return
    
    # If camera_index is not in available cameras, use the first one
    if camera_index not in available_cameras:
        camera_index = available_cameras[0]
        print(f"Selected camera index {camera_index}")
    
    print("Initializing webcam...")
    
    # Initialize webcam with retry mechanism
    video_capture = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open camera. Please check your camera permissions:")
        print("1. Go to System Settings → Privacy & Security → Camera")
        print("2. Ensure that your terminal application has permission to access the camera")
        print("3. Try running the program again")
        return
    
    # Give the camera a moment to initialize
    time.sleep(1)
    
    print("Press 'q' to quit")
    print("Press 's' to save the current frame")
    print("Press 'c' to change camera")
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        # Check if frame was successfully captured
        if not ret or frame is None:
            print("Failed to capture frame. Check camera connection and permissions.")
            time.sleep(0.5)
            continue
            
        # Detect faces
        try:
            face_imgs, face_coords, gray = detect_faces(frame)
            
            # Draw rectangles around faces
            for (x, y, w, h) in face_coords:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Register Face', frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Continue to next frame
            pass
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Change camera
            video_capture.release()
            camera_index = (camera_index + 1) % (max(available_cameras) + 1)
            if camera_index not in available_cameras:
                camera_index = available_cameras[0]
            
            print(f"Switching to camera {camera_index}")
            video_capture = cv2.VideoCapture(camera_index)
            
            if not video_capture.isOpened():
                print(f"Failed to open camera {camera_index}, trying another one")
                continue
        elif key == ord('s'):
            if not 'face_coords' in locals() or not face_coords:
                print("No face detected in the frame. Try again.")
                continue
            elif len(face_coords) > 1:
                print("Multiple faces detected. Please ensure only one person is in the frame.")
                continue
            
            # Get name from user
            name = input("Enter the name of the person: ")
            
            # Save the first detected face
            face_img = face_imgs[0]
            save_face_data(name, face_img)
            print(f"Face registered successfully for {name}")
    
    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use camera index 0 (usually the built-in webcam)
    register_face(camera_index=0) 