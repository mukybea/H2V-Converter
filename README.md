# Horizontal-to-Vertical Video Converter (H2VConverter)

## Overview

The H2VConverter is a tool that converts horizontal videos into a vertical format suitable for platforms like TikTok, Instagram Reels, and YouTube Shorts, etc. It analyzes video content, detects scenes, identifies key objects and speakers, and automatically crops and pans a given videos.

## Features

* **Smart Cropping and Panning:** Automatically crops and pans horizontal videos to create visually appealing vertical videos.
* **Scene Detection:** Detects scene changes to optimize cropping and panning.
* **Speech Detection:** Uses Whisper to detect speech segments and prioritize active speakers.
* **Object Detection:** Employs the recent YOLO v12 to identify and track important objects.
* **Face and Pose Detection:** Utilizes MediaPipe to detect faces and human poses for focusing on people.

## Installation

Quick note: Due to version compatibility with MediaPipe installation a Python 3.11 was used.

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:mukybea/H2V-Converter.git
    cd H2V-Converter
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    conda create -n myenv python=3.11
    conda activate myenv
    ```

3.  **Install the Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

```bash
python h2v_converter.py <input_video> [--output-dir <output_directory>] [--ratio <width:height>] [--model_size <model_size>]
```
### Arguments:
```bash
--input_video (required): Path to the input horizontal video file.
--output-dir (optional): Path to the output directory. Default is output.
--ratio (optional): Target aspect ratio for the output video in the format width:height. Default is 9:16.
--model_size (optional): Size of the Whisper model to use for speech detection.  Options are tiny, base, small, medium, and large. Default is base.
```

Example: 
```bash
python h2v_converter.py input.mp4 --output-dir output_videos --ratio 9:16 --model_size base
```

### Output:

The code generates two output files: 
1. `final_vertical.mp4` (the output vertical video)
2. `focus_points.json` (a JSON file containing the focus points used for cropping and panning)

## Code Structure
- main.py
  - H2VConverter
    - \_\_init\_\_ _Initializes the converter, loads models, and sets up video capture_
    - calculate_target_dimensions: _compute the dimensions of the output vertical video_
    - detect_scenes: _detects scene changes in the input video_
    - extract_audio: _extracts the audio track from the given video_
      - detect_speech_segments: _detects speech segments in the audio track_
        - is_frame_in_speech_segment: Checks if a frame is within a speech segment.
    - _get_face_detections: _detect faces of persons to identify focus points_
    - _get_pose_detections: _detect poses of persons to identify focus points_
    - detect_active_speaker: _detects the active speaker in a frame_
    - detect_important_objects: _detects important objects in a frame_
    - _get_focus_object: _determines the focus object_
    - process_scene: _processes a scene and calculates focus points_
    - smooth_focus_points: _smooths focus points to prevent jittery movement_
    - generate_output_video: _generates the output vertical video_
