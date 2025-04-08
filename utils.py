import os
import cv2
import numpy as np
from datetime import datetime
import csv
import pickle
import sys

# Check if face module is available
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    FACE_MODULE_AVAILABLE = True
except (AttributeError, ModuleNotFoundError):
    print("WARNING: OpenCV face module not available. Please install opencv-contrib-python:")
    print("pip install opencv-contrib-python")
    FACE_MODULE_AVAILABLE = False

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['known_faces', 'attendance_records', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def get_face_detector():
    """Get face detector model"""
    # Use OpenCV's built-in Haar cascade classifier
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise Exception("Failed to load built-in cascade classifier")
        return face_cascade
    except Exception as e:
        print(f"Error loading built-in cascade classifier: {e}")
        
        # As a fallback, try to find a downloaded model
        face_cascade_path = os.path.join('models', 'haarcascade_frontalface_default.xml')
        if os.path.exists(face_cascade_path):
            return cv2.CascadeClassifier(face_cascade_path)
        else:
            print("WARNING: Could not load any face detection model!")
            # Return a placeholder classifier that will be empty
            return cv2.CascadeClassifier()

def load_known_faces():
    """Load known face data from the known_faces directory"""
    if not FACE_MODULE_AVAILABLE:
        print("ERROR: OpenCV face module not available. Cannot load face recognizer.")
        print("Please install opencv-contrib-python and try again.")
        print("pip install opencv-contrib-python")
        sys.exit(1)
    
    known_face_data = []
    known_face_names = []
    
    # LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Check if we have a trained model
    model_path = os.path.join('models', 'face_recognizer.yml')
    if os.path.exists(model_path):
        recognizer.read(model_path)
        
        # Load labels
        labels_path = os.path.join('models', 'face_labels.pkl')
        if os.path.exists(labels_path):
            with open(labels_path, 'rb') as f:
                label_dict = pickle.load(f)
                # Invert the dictionary to get id->name mapping
                id_name_dict = {v: k for k, v in label_dict.items()}
        else:
            id_name_dict = {}
    else:
        id_name_dict = {}
    
    return recognizer, id_name_dict

def save_face_data(name, face_img):
    """Save face data for training"""
    if not FACE_MODULE_AVAILABLE:
        print("ERROR: OpenCV face module not available. Cannot save face data.")
        print("Please install opencv-contrib-python and try again.")
        print("pip install opencv-contrib-python")
        return
    
    # Create directory for this person if it doesn't exist
    person_dir = os.path.join('known_faces', name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    # Save the face image
    count = len(os.listdir(person_dir))
    filename = os.path.join(person_dir, f"{count}.jpg")
    cv2.imwrite(filename, face_img)
    
    # Update the model (train with all available faces)
    train_face_recognizer()

def train_face_recognizer():
    """Train the face recognizer with all available face images"""
    if not FACE_MODULE_AVAILABLE:
        print("ERROR: OpenCV face module not available. Cannot train face recognizer.")
        print("Please install opencv-contrib-python and try again.")
        print("pip install opencv-contrib-python")
        return
    
    faces = []
    labels = []
    label_dict = {}
    current_id = 0
    
    # Iterate through all person directories
    for person_name in os.listdir('known_faces'):
        person_dir = os.path.join('known_faces', person_name)
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
        model_path = os.path.join('models', 'face_recognizer.yml')
        recognizer.write(model_path)
        
        # Save the label dictionary
        labels_path = os.path.join('models', 'face_labels.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(label_dict, f)
        
        print(f"Face recognition model trained with {len(faces)} images of {len(label_dict)} people")
    else:
        print("No faces found for training")

def mark_attendance(name):
    """Mark attendance for a person"""
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H:%M:%S')
    
    filename = os.path.join('attendance_records', f'attendance_{date}.csv')
    
    # Check if file exists and create header if needed
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Time'])
        writer.writerow([name, time])

def detect_faces(frame):
    """Detect faces in a frame using Haar cascade"""
    if frame is None:
        raise ValueError("Frame is None, cannot detect faces")
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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
    
    face_imgs = []
    face_coords = []
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = gray[y:y+h, x:x+w]
        face_imgs.append(face_img)
        face_coords.append((x, y, w, h))
    
    return face_imgs, face_coords, gray 