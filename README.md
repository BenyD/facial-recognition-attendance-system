# Facial Recognition Attendance System

A Python-based facial recognition attendance system that uses computer vision to track attendance.

## Setup

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your configuration (see .env.example)

## Usage

1. To register new faces:

   ```bash
   python register.py
   ```

2. To start the attendance system:
   ```bash
   python attendance.py
   ```

## Features

- Face detection and recognition
- Real-time attendance tracking
- CSV export of attendance records
- Simple and intuitive interface

## Project Structure

- `attendance.py` - Main attendance system
- `register.py` - Face registration utility
- `utils.py` - Helper functions
- `known_faces/` - Directory for storing known face encodings
- `attendance_records/` - Directory for storing attendance records
