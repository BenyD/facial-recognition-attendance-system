# Facial Recognition Attendance System

A Python-based facial recognition attendance system that uses computer vision to track attendance.

## Setup

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the `.env` file with your settings:

   ```
   # Camera settings (0=external camera, 1=built-in camera)
   CAMERA_INDEX=1

   # Face recognition settings
   CONFIDENCE_THRESHOLD=70

   # Attendance settings
   ATTENDANCE_COOLDOWN=60
   ```

## Usage

Run the application:

```bash
python app.py
```

This will display a menu with options to:

1. Register a new face
2. Take attendance
3. Exit

## Features

- Face detection and recognition using OpenCV
- Real-time attendance tracking with confirmation
- CSV export of attendance records by date
- Easy face registration system
- Smart camera selection with fallback options
- Configurable settings via environment variables

## Project Structure

- `app.py` - Main application with all functionality
- `.env` - Configuration file for settings
- `known_faces/` - Directory for storing known face images
- `attendance_records/` - Directory for storing attendance CSV files
- `models/` - Directory for storing trained face recognition models

## Customization

You can customize the system by modifying the `.env` file:

- Change `CAMERA_INDEX` to use a different camera
- Adjust `CONFIDENCE_THRESHOLD` for stricter/looser face matching
- Modify `ATTENDANCE_COOLDOWN` to control how frequently attendance is recorded
