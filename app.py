import cv2
import os
import numpy as np
import time
import csv
import pickle
import sys
import signal
from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables
load_dotenv()

def get_env_var(name, default, type_func=str):
    """Get environment variable and convert to specified type, handling comments"""
    value = os.getenv(name, default)
    # Remove any comments from the value
    value = value.split('#')[0].strip()
    return type_func(value)

# Configure settings from environment variables
CAMERA_INDEX = get_env_var("CAMERA_INDEX", "1", int)  # Default to 1 (built-in camera)
CONFIDENCE_THRESHOLD = get_env_var("CONFIDENCE_THRESHOLD", "70", float)
ATTENDANCE_COOLDOWN = get_env_var("ATTENDANCE_COOLDOWN", "60", int)
KNOWN_FACES_DIR = get_env_var("KNOWN_FACES_DIR", "known_faces", str)
ATTENDANCE_DIR = get_env_var("ATTENDANCE_DIR", "attendance_records", str)
MODELS_DIR = get_env_var("MODELS_DIR", "models", str)
FRAME_SKIP = get_env_var("FRAME_SKIP", "2", int)  # Process every nth frame
FACE_DETECTION_SCALE = get_env_var("FACE_DETECTION_SCALE", "0.5", float)  # Scale factor for face detection

# Global variables for cleanup and performance
video_capture = None
windows = set()
face_detector = None
last_frame_time = 0
frame_count = 0
fps = 0

def cleanup():
    """Cleanup function to release resources"""
    global video_capture, face_detector
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
    for window in windows:
        cv2.destroyWindow(window)
    cv2.destroyAllWindows()
    face_detector = None
    print("\nResources cleaned up successfully.")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal. Cleaning up...")
    cleanup()
    sys.exit(0)

# Register signal handler for graceful termination
signal.signal(signal.SIGINT, signal_handler)

# Check if face module is available
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    FACE_MODULE_AVAILABLE = True
except (AttributeError, ModuleNotFoundError):
    print("WARNING: OpenCV face module not available. Please install opencv-contrib-python:")
    print("pip install opencv-contrib-python")
    FACE_MODULE_AVAILABLE = False

# ===== UTILITY FUNCTIONS =====

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [KNOWN_FACES_DIR, ATTENDANCE_DIR, MODELS_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

@lru_cache(maxsize=1)
def get_face_detector():
    """Get face detector model with caching"""
    # Use OpenCV's built-in Haar cascade classifier
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise Exception("Failed to load built-in cascade classifier")
        return face_cascade
    except Exception as e:
        print(f"Error loading built-in cascade classifier: {e}")
        
        # As a fallback, try to find a downloaded model
        face_cascade_path = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
        if os.path.exists(face_cascade_path):
            return cv2.CascadeClassifier(face_cascade_path)
        else:
            print("WARNING: Could not load any face detection model!")
            # Return a placeholder classifier that will be empty
            return cv2.CascadeClassifier()

def load_known_faces():
    """Load known face data with caching"""
    if not FACE_MODULE_AVAILABLE:
        print("ERROR: OpenCV face module not available. Cannot load face recognizer.")
        print("Please install opencv-contrib-python and try again.")
        print("pip install opencv-contrib-python")
        sys.exit(1)
    
    # LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Check if we have a trained model
    model_path = os.path.join(MODELS_DIR, 'face_recognizer.yml')
    if os.path.exists(model_path):
        try:
            recognizer.read(model_path)
            
            # Load labels
            labels_path = os.path.join(MODELS_DIR, 'face_labels.pkl')
            if os.path.exists(labels_path):
                with open(labels_path, 'rb') as f:
                    label_dict = pickle.load(f)
                    # Invert the dictionary to get id->name mapping
                    id_name_dict = {v: k for k, v in label_dict.items()}
                    print(f"Loaded {len(id_name_dict)} known faces")
            else:
                id_name_dict = {}
                print("No face labels found")
        except Exception as e:
            print(f"Error loading face recognition model: {e}")
            id_name_dict = {}
    else:
        id_name_dict = {}
        print("No face recognition model found")
    
    return recognizer, id_name_dict

def save_face_data(name, face_img):
    """Save face data for training"""
    if not FACE_MODULE_AVAILABLE:
        print("ERROR: OpenCV face module not available. Cannot save face data.")
        print("Please install opencv-contrib-python and try again.")
        print("pip install opencv-contrib-python")
        return False
    
    try:
        # Create directory for this person if it doesn't exist
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Save the face image
        count = len(os.listdir(person_dir))
        filename = os.path.join(person_dir, f"{count}.jpg")
        cv2.imwrite(filename, face_img)
        
        # Update the model (train with all available faces)
        train_face_recognizer()
        return True
    except Exception as e:
        print(f"Error saving face data: {e}")
        return False

def train_face_recognizer():
    """Train the face recognizer with all available face images"""
    if not FACE_MODULE_AVAILABLE:
        print("ERROR: OpenCV face module not available. Cannot train face recognizer.")
        print("Please install opencv-contrib-python and try again.")
        print("pip install opencv-contrib-python")
        return
    
    try:
        faces = []
        labels = []
        label_dict = {}
        current_id = 0
        
        # Iterate through all person directories
        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            if os.path.isdir(person_dir):
                # Assign a label ID to this person
                if person_name not in label_dict:
                    label_dict[person_name] = current_id
                    current_id += 1
                
                person_id = label_dict[person_name]
                
                # Load all face images for this person
                for img_name in os.listdir(person_dir):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(person_dir, img_name)
                        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if face_img is not None:
                            # Add to training data
                            faces.append(face_img)
                            labels.append(person_id)
        
        if faces and labels:
            # Create and train LBPH Face Recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(labels))
            
            # Save the model
            model_path = os.path.join(MODELS_DIR, 'face_recognizer.yml')
            recognizer.write(model_path)
            
            # Save the label dictionary
            labels_path = os.path.join(MODELS_DIR, 'face_labels.pkl')
            with open(labels_path, 'wb') as f:
                pickle.dump(label_dict, f)
            
            print(f"Face recognition model trained with {len(faces)} images of {len(label_dict)} people")
        else:
            print("No faces found for training")
    except Exception as e:
        print(f"Error training face recognizer: {e}")

def mark_attendance(name):
    """Mark attendance for a person"""
    try:
        date = datetime.now().strftime('%Y-%m-%d')
        time = datetime.now().strftime('%H:%M:%S')
        
        filename = os.path.join(ATTENDANCE_DIR, f'attendance_{date}.csv')
        
        # Check if file exists and create header if needed
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Name', 'Time'])
            writer.writerow([name, time])
        return True
    except Exception as e:
        print(f"Error marking attendance: {e}")
        return False

def detect_faces(frame):
    """Detect faces in a frame using Haar cascade with optimizations"""
    if frame is None:
        raise ValueError("Frame is None, cannot detect faces")
        
    try:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
        
        # Convert to grayscale
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Get face detector
        face_detector = get_face_detector()
        
        # Detect faces
        if face_detector.empty():
            print("WARNING: Face detector is empty, cannot detect faces")
            return [], [], gray
        
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Scale back the coordinates
        faces = [(int(x/FACE_DETECTION_SCALE), int(y/FACE_DETECTION_SCALE), 
                 int(w/FACE_DETECTION_SCALE), int(h/FACE_DETECTION_SCALE)) 
                for (x, y, w, h) in faces]
        
        face_imgs = []
        face_coords = []
        
        for (x, y, w, h) in faces:
            # Extract face region from original frame
            face_img = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            face_imgs.append(face_img)
            face_coords.append((x, y, w, h))
        
        return face_imgs, face_coords, gray
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return [], [], frame

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

def initialize_camera(camera_index, available_cameras=None):
    """Initialize a camera with the given index, fallback to available ones if needed"""
    global video_capture
    
    if available_cameras is None:
        available_cameras = list_available_cameras()
    
    if not available_cameras:
        print("No cameras found!")
        return None
    
    # If camera_index is not in available cameras, use the first one
    if camera_index not in available_cameras:
        camera_index = available_cameras[0]
        print(f"Requested camera index {camera_index} not available. Using camera {available_cameras[0]} instead.")
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open camera. Please check your camera permissions:")
        print("1. Go to System Settings → Privacy & Security → Camera")
        print("2. Ensure that your terminal application has permission to access the camera")
        print("3. Try running the program again")
        return None
    
    # Give the camera a moment to initialize
    time.sleep(1)
    
    return video_capture

def show_window(title, frame):
    """Show a window and track it for cleanup"""
    cv2.imshow(title, frame)
    windows.add(title)

# ===== MAIN FUNCTIONS =====

def register_face_mode():
    """Register a new face"""
    print("\n== Face Registration Mode ==")
    
    try:
        # Create necessary directories
        create_directories()
        
        # List available cameras
        available_cameras = list_available_cameras()
        if not available_cameras:
            print("No cameras found!")
            return
        
        # Initialize camera
        video_capture = initialize_camera(CAMERA_INDEX, available_cameras)
        if video_capture is None:
            return
        
        print("\nInstructions:")
        print("Press 'q' to quit")
        print("Press 's' to save the current frame")
        print("Press 'c' to change camera")
        print("\nPosition your face in the frame and press 's' when ready")
        
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
                show_window('Register Face', frame)
            except Exception as e:
                print(f"Error processing frame: {e}")
                show_window('Register Face', frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Change camera
                video_capture.release()
                new_index = (CAMERA_INDEX + 1) % (max(available_cameras) + 1)
                if new_index not in available_cameras:
                    new_index = available_cameras[0]
                
                print(f"Switching to camera {new_index}")
                video_capture = cv2.VideoCapture(new_index)
                
                if not video_capture.isOpened():
                    print(f"Failed to open camera {new_index}, trying another one")
                    continue
            elif key == ord('s'):
                if 'face_coords' not in locals() or not face_coords:
                    print("No face detected in the frame. Try again.")
                    continue
                elif len(face_coords) > 1:
                    print("Multiple faces detected. Please ensure only one person is in the frame.")
                    continue
                
                # Get name from user
                name = input("\nEnter the name of the person: ")
                
                # Save the first detected face
                face_img = face_imgs[0]
                if save_face_data(name, face_img):
                    print(f"\nFace registered successfully for {name}")
                    print("Press 'q' to return to main menu or 's' to register another face")
                else:
                    print("Failed to register face. Please try again.")
    
    except Exception as e:
        print(f"Error in face registration: {e}")
    finally:
        # Cleanup
        cleanup()
        print("\nReturning to main menu...")

def attendance_mode():
    """Run the attendance system"""
    print("\n== Attendance Mode ==")
    
    try:
        # Create necessary directories
        create_directories()
        
        # Load face recognizer and label mapping
        recognizer, id_name_dict = load_known_faces()
        
        if not id_name_dict:
            print("No registered faces found. Please register faces first using register mode.")
            return
        
        # List available cameras
        available_cameras = list_available_cameras()
        if not available_cameras:
            print("No cameras found!")
            return
        
        # Initialize camera
        video_capture = initialize_camera(CAMERA_INDEX, available_cameras)
        if video_capture is None:
            return
        
        # Dictionary to keep track of when we last saw a person
        last_seen = {}
        # Flag to indicate if we're in confirmation mode after detection
        confirmation_mode = False
        # Last person detected
        last_detected_person = None
        
        print("\nInstructions:")
        print("Press 'q' to quit")
        print("Press 'c' to change camera")
        print("\nStarting attendance system...")
        
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            
            # Check if frame was successfully captured
            if not ret or frame is None:
                print("Failed to capture frame. Check camera connection and permissions.")
                time.sleep(0.5)
                continue
            
            # Create a copy of frame for displaying confirmation screen
            display_frame = frame.copy()
            
            if confirmation_mode and last_detected_person:
                # Show confirmation screen with options
                cv2.putText(display_frame, f"Attendance marked for: {last_detected_person}", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'q' to quit", 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press 'c' to change camera", 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press any other key to continue scanning", 
                           (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Display the frame with confirmation options
                show_window('Attendance System', display_frame)
                
                # Wait for key press (this is a blocking wait)
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Change camera
                    video_capture.release()
                    new_index = (CAMERA_INDEX + 1) % (max(available_cameras) + 1)
                    if new_index not in available_cameras:
                        new_index = available_cameras[0]
                    
                    print(f"Switching to camera {new_index}")
                    video_capture = cv2.VideoCapture(new_index)
                    
                    if not video_capture.isOpened():
                        print(f"Failed to open camera {new_index}, trying another one")
                        continue
                
                # Exit confirmation mode
                confirmation_mode = False
                continue
            
            try:
                # Detect faces in the frame
                face_imgs, face_coords, gray = detect_faces(frame)
                
                # Process each detected face
                for i, ((x, y, w, h), face_img) in enumerate(zip(face_coords, face_imgs)):
                    # Try to recognize the face
                    try:
                        # Predict the person ID and confidence
                        person_id, confidence = recognizer.predict(face_img)
                        
                        # Lower confidence value means better match in LBPH
                        if confidence < CONFIDENCE_THRESHOLD:  # Confidence threshold
                            name = id_name_dict.get(person_id, "Unknown")
                            confidence_text = f"{100 - confidence:.1f}%"
                            
                            # Check if we should mark attendance for this person
                            current_time = datetime.now()
                            if name not in last_seen or (current_time - last_seen[name]).total_seconds() > ATTENDANCE_COOLDOWN:
                                if mark_attendance(name):
                                    last_seen[name] = current_time
                                    print(f"Attendance marked for {name}")
                                    
                                    # Enter confirmation mode
                                    confirmation_mode = True
                                    last_detected_person = name
                        else:
                            name = "Unknown"
                            confidence_text = ""
                    except:
                        name = "Unknown"
                        confidence_text = ""
                    
                    # Draw rectangle around the face
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, red for unknown
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Display name and confidence
                    if name != "Unknown":
                        cv2.rectangle(frame, (x, y+h), (x+w, y+h+35), color, cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        display_text = f"{name} {confidence_text}"
                        cv2.putText(frame, display_text, (x+6, y+h+25), font, 0.6, (255, 255, 255), 1)
                
                # Display the resulting frame with instructions
                if not confirmation_mode:
                    cv2.putText(frame, "Press 'q' to quit", (10, 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame, "Press 'c' to change camera", (10, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    show_window('Attendance System', frame)
                              
            except Exception as e:
                print(f"Error processing frame: {e}")
                show_window('Attendance System', frame)
            
            if not confirmation_mode:
                # Process key presses (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Change camera
                    video_capture.release()
                    new_index = (CAMERA_INDEX + 1) % (max(available_cameras) + 1)
                    if new_index not in available_cameras:
                        new_index = available_cameras[0]
                    
                    print(f"Switching to camera {new_index}")
                    video_capture = cv2.VideoCapture(new_index)
                    
                    if not video_capture.isOpened():
                        print(f"Failed to open camera {new_index}, trying another one")
                        continue
    
    except Exception as e:
        print(f"Error in attendance mode: {e}")
    finally:
        # Cleanup
        cleanup()
        print("\nReturning to main menu...")

def display_menu():
    """Display the main menu and handle user choice"""
    while True:
        print("\n=== Facial Recognition Attendance System ===")
        print("1. Register a new face")
        print("2. Take attendance")
        print("3. Exit")
        
        try:
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                register_face_mode()
            elif choice == '2':
                attendance_mode()
            elif choice == '3':
                print("\nThank you for using the Facial Recognition Attendance System!")
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Exiting...")
            cleanup()
            break
        except Exception as e:
            print(f"Error in menu: {e}")

if __name__ == "__main__":
    try:
        display_menu()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        cleanup() 