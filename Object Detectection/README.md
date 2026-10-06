# NMDC Stage 1 - Open-Vocabulary RGB Object Detection

This package is a ready-to-run starter for detecting a broad set of potential
objects/obstacles in an enhanced RGB mining-road video.

It uses Ultralytics YOLO-World. The model supports open-vocabulary detection
through text prompts, so the detector can be configured with classes such as
person, vehicle, truck, excavator, dump truck, rock, debris and road obstacle.

## Folder

- `stage1_openvocab_detect.py` - main program
- `requirements.txt` - Python dependencies
- `outputs/` - generated videos

## Installation

Recommended Python environment:

```powershell
python -m pip install -r requirements.txt
```

## Run

From this folder:

```powershell
python stage1_openvocab_detect.py --source "YOUR_ENHANCED_VIDEO.mp4" --side-by-side
```

Example for the NMDC project:

```powershell
python stage1_openvocab_detect.py --source "outputs/advanced_enhanced.mp4" --side-by-side
```

The output is:

```text
outputs/rgb_openvocab_detected.mp4
```

## Notes

- On the first run, the YOLO-World weights may be downloaded automatically by Ultralytics.
- The prompts in the Python file are starting prompts for a mining-road prototype.
- Open-vocabulary detection does not guarantee detection of every unknown object.
  For a safety-critical system, add free-space/depth estimation, tracking,
  thermal fusion and validation on mine-specific data.
- Confidence and image size can be changed, for example:
  `--conf 0.20 --imgsz 640`

## Project progression

Enhanced RGB -> Open-vocabulary detection -> Tracking -> Thermal/RGB fusion
-> Depth/distance estimation -> RTK-GNSS geo-tagging -> hazard/risk logic.
