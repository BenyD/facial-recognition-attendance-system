import cv2
import time
from utils import create_directories, load_known_faces, detect_faces, mark_attendance
from datetime import datetime

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

def main(camera_index=0):
    # Create necessary directories
    create_directories()
    
    # Load face recognizer and label mapping
    recognizer, id_name_dict = load_known_faces()
    
    if not id_name_dict:
        print("No registered faces found. Please register faces first using register.py")
        return
    
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
    
    # Initialize webcam
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
    
    # Dictionary to keep track of when we last saw a person
    last_seen = {}
    # Flag to indicate if we're in confirmation mode after detection
    confirmation_mode = False
    # Last person detected
    last_detected_person = None
    
    print("Starting attendance system...")
    print("Press 'q' to quit")
    print("Press 'c' to change camera")
    
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
            cv2.imshow('Attendance System', display_frame)
            
            # Wait for key press (this is a blocking wait)
            key = cv2.waitKey(0) & 0xFF
            
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
                    if confidence < 70:  # Confidence threshold
                        name = id_name_dict.get(person_id, "Unknown")
                        confidence_text = f"{100 - confidence:.1f}%"
                        
                        # Check if we should mark attendance for this person
                        current_time = datetime.now()
                        if name not in last_seen or (current_time - last_seen[name]).total_seconds() > 60:
                            mark_attendance(name)
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
                cv2.imshow('Attendance System', frame)
                          
        except Exception as e:
            print(f"Error processing frame: {e}")
            cv2.imshow('Attendance System', frame)
        
        if not confirmation_mode:
            # Process key presses (non-blocking)
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
    
    # Release the webcam and close windows
    print("Gracefully shutting down...")
    video_capture.release()
    cv2.destroyAllWindows()
    print("Application terminated")

if __name__ == "__main__":
    # Use camera index 0 (usually the built-in webcam)
    main(camera_index=0) 