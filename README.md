# Mining_HEMM_Safety_-_Visibility_Enhancement_System
This is the first module of the proposed HEMM system.

## Goal

Take RGB camera video as input and improve local contrast/visibility before object detection.

### Pipeline

RGB video
-> HSV conversion
-> CLAHE
-> optional percentile color balance
-> enhanced video
-> object detection (Task 2)

## Modes

### Baseline CIMFR-style enhancement

```bash
python dehaze.py --source fog_video.mp4 --mode clahe --side-by-side
```

### Adaptive enhancement

```bash
python dehaze.py --source fog_video.mp4 --mode adaptive --side-by-side
```

### Webcam

```bash
python dehaze.py --source 0 --mode adaptive --side-by-side
```

## Output

Enhanced video is saved by default to:

`outputs/enhanced_output.mp4`

## Important engineering note

The "proxy visibility" displayed by this program is an image-quality indicator, not a calibrated atmospheric visibility value in metres.

For the final HEMM system, add a physical forward-scatter visibility sensor and correlate it with image metrics.

## Why this is better than the original code

1. Uses vectorized operations instead of Python pixel-by-pixel loops.
2. Processes complete video streams.
3. Supports webcam or recorded fog video.
4. Shows original vs enhanced output.
5. Has an adaptive enhancement mode.
6. Measures processing FPS.
7. Produces a saved video that can be used as input to Task 2 object detection.

## Next step

The output of this program becomes the input to:

`Task 2 — RGB/Thermal/Radar Object Detection and Tracking`.
