# YOLO / Mask R-CNN Annotator

## Overview
The **YOLO / Mask R-CNN Annotator** is a powerful tool designed for annotating images and videos with bounding boxes and polygons, making it suitable for deep learning tasks such as object detection and instance segmentation. The tool allows you to create, edit, and export annotations in formats compatible with YOLO,and Mask RCNN JSON. It also supports AI-assisted pre-labeling using pretrained YOLOv11 and provides functionalities to train custom models.

> **Note:** Some features are still under development and may not be fully integrated.

---

## Features
- **Image Annotation**
  - Draw bounding boxes (BBox) and polygons for object segmentation.
  - Save and load annotations in multiple formats.
  - Undo/redo and edit annotations.
- **Video Annotation** *(Requires OpenCV)*
  - Extract frames from videos for annotation.
- **Class Management**
  - Add, remove, and manage object classes.
  - Select object classes via the sidebar.
- **Navigation & Viewing**
  - Zoom and pan within images.
  - Navigate through images using a file tree.
- **Annotation Export**
  - YOLO format.
  - Mask R-CNN
- **AI-Assisted Pre-Annotation** *(YOLOv11-based, Requires Ultralytics)*
  - Use a pre-trained YOLOv8 model to auto-detect objects and generate annotations.
- **Custom Model Training** *(Requires Ultralytics)*
  - Train a custom YOLO model with your dataset.
- **Quality Control Tools**
  - Detect overlapping bounding boxes.
- **Project Management**
  - Save and load annotation projects.
  - Auto-save functionality.

---

## Installation

### Prerequisites
Ensure you have **Python 3.7+** installed on your system.

### Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** If you plan to use AI-assisted annotation or train a custom model, you need to install **Ultralytics**.
```bash
pip install ultralytics
```

For video annotation support, install OpenCV:
```bash
pip install opencv-python
```

---

## Usage

### Running the Application
Run the main script to start the annotator:
```bash
python main.py
```

### Loading Images or Videos
- Click **File → Load Image Folder** to select a folder containing images.
- If OpenCV is installed, click **File → Load Video** to extract frames from a video.

### Annotating Objects
- **Bounding Box Mode:** Click and drag to create a rectangular annotation.
- **Polygon Mode:** Click to create polygon points, then double-click to close the shape.
- Use the **Edit Mode** to modify existing annotations.

### Exporting Annotations
Go to **Export** in the menu to save annotations in:
- YOLO format (fully implemented).
- Mask R-CNN

### AI-Assisted Pre-Annotation
To use YOLOv8 for automatic annotation:
1. Install **Ultralytics** (`pip install ultralytics`).
2. Click **Tools → AI Pre-label (YOLOv8)**.
3. The pre-trained YOLOv8 model will detect objects and generate annotations automatically.

### Training a Custom Model
1. Install **Ultralytics** (`pip install ultralytics`).
2. Prepare a dataset in YOLO format and a `dataset.yaml` file.
3. Click **Tools → Train Custom Model** and select your dataset.
4. Choose the number of epochs and start training and add others parameters.

> **Note:** Training is done using the default YOLOv11 model.

---

## Keyboard Shortcuts

The following key bindings are available in the Advanced YOLO / Mask R-CNN Annotator:

| **Shortcut**           | **Function**                                   |
|------------------------|------------------------------------------------|
| **N** or **n**         | Load the next image                            |
| **P** or **p**         | Load the previous image                        |
| **D** or **d**         | Delete the selected annotation                 |
| **Z** or **z**         | Undo the last action                           |
| **Y** or **y**         | Redo the last undone action                    |
| **Ctrl + S**           | Save the current project                       |
| **Ctrl + Z**           | Undo (alternative shortcut)                    |
| **Ctrl + Y**           | Redo (alternative shortcut)                    |
| **Double-click**       | (In polygon mode) Finish drawing polygon       |
| **Right-click & drag** | Pan the image                                  |
| **Mouse wheel**        | Zoom in/out the image                          |

---

## Known Issues & Future Improvements
- **Pascal VOC and COCO export formats are not fully implemented.**
- **Dataset splitting functionality is not yet available.**
- **Polygon editing needs improvements.**

---

## License
### Public Use License (Attribution Required)  

This project is made available to the public for use, modification, and distribution under the following conditions:  

- **Attribution** - Any use, modification, or distribution of this project must include appropriate credit to the original author. This includes a clear reference to the original source in any derivative works or public distributions.  
- **Modification and Redistribution** - Users are permitted to modify and redistribute the project, provided that attribution is maintained and any significant changes are noted.  
- **No Warranty or Liability** - This project is provided "as-is" without any warranties, expressed or implied. The author assumes no responsibility for any issues arising from its use.  
- **Fair Use and Ethical Application** - This project must not be used for unlawful, unethical, or harmful purposes.  

By using this project, you agree to adhere to these terms. 
---

## Acknowledgments
- **Ultralytics** for the YOLOv11 framework.
- **OpenCV** for video processing support.
- **Pillow & Tkinter** for GUI development."# All-in-One-Label" 
