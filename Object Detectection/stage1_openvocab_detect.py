import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# OBJECT VOCABULARY
# ============================================================

DEFAULT_CLASSES = [
    "person",
    "vehicle",
    "truck",
    "car",
    "bus",
    "excavator",
    "dump truck",
    "loader",
    "construction equipment",
    "rock",
    "stone",
    "debris",
    "road obstacle",
    "animal",
]


# ============================================================
# SOURCE CONVERTER
# ============================================================

def open_source(source):
    """
    Convert a numeric camera source such as '0' or '1'
    into an integer. Otherwise keep it as a video path.
    """

    try:
        return int(source)
    except ValueError:
        return source


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENTS
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "NMDC Stage 1 - Open-vocabulary obstacle "
            "detection using YOLO-World"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Camera number (0/1) or video filename"
    )

    parser.add_argument(
        "--output",
        default="outputs/rgb_openvocab_detected.mp4",
        help="Output video filename"
    )

    parser.add_argument(
        "--model",
        default="yolov8s-world.pt",
        help="YOLO-World model weights"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size"
    )

    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Show input and detection side-by-side"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CONVERT SOURCE
    # --------------------------------------------------------

    source = open_source(args.source)

    # --------------------------------------------------------
    # LOAD YOLO-WORLD
    # --------------------------------------------------------

    print("\nLoading YOLO-World model...")

    model = YOLO(args.model)

    model.set_classes(DEFAULT_CLASSES)

    print("\nCustom object vocabulary loaded:")
    print(", ".join(DEFAULT_CLASSES))

    # --------------------------------------------------------
    # OPEN CAMERA OR VIDEO
    # --------------------------------------------------------

    if isinstance(source, int):

        print(f"\nOpening camera {source}...")

        cap = cv2.VideoCapture(
            source,
            cv2.CAP_DSHOW
        )

        # Camera resolution
        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        # Camera FPS
        cap.set(
            cv2.CAP_PROP_FPS,
            30
        )

        live_camera = True

    else:

        print(f"\nOpening video: {source}")

        cap = cv2.VideoCapture(
            source
        )

        live_camera = False

    # --------------------------------------------------------
    # CHECK SOURCE
    # --------------------------------------------------------

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open source: {args.source}"
        )

    # --------------------------------------------------------
    # VIDEO PROPERTIES
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 1:

        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # --------------------------------------------------------
    # PRINT SOURCE INFORMATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NMDC STAGE 1 - OPEN-VOCABULARY RGB OBJECT DETECTION")
    print("=" * 70)

    if live_camera:
        print(f"Source     : Camera {source}")
    else:
        print(f"Source     : {args.source}")

    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.2f}")

    if live_camera:

        print("Mode       : LIVE CAMERA")

    else:

        print(f"Frames     : {total}")

        if fps > 0:
            print(
                f"Duration   : {total / fps:.2f} seconds"
            )

    print("=" * 70)

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    out_width = (
        width * 2
        if args.side_by_side
        else width
    )

    writer = cv2.VideoWriter(

        str(output_path),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            out_width,
            height
        )
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create output video."
        )

    # --------------------------------------------------------
    # PROCESSING
    # --------------------------------------------------------

    frame_no = 0

    start_time = time.perf_counter()

    while True:

        # ====================================================
        # READ FRAME
        # ====================================================

        ok, frame = cap.read()

        if not ok:

            print(
                "\nCould not read next frame."
            )

            break

        # ====================================================
        # YOLO-WORLD DETECTION
        # ====================================================

        results = model.predict(

            source=frame,

            conf=args.conf,

            imgsz=args.imgsz,

            verbose=False
        )

        result = results[0]

        # ====================================================
        # DRAW DETECTIONS
        # ====================================================

        annotated = result.plot()

        # ====================================================
        # COUNT OBJECTS
        # ====================================================

        if result.boxes is None:

            object_count = 0

        else:

            object_count = len(
                result.boxes
            )

        # ====================================================
        # PROCESSING FPS
        # ====================================================

        frame_no += 1

        elapsed = (
            time.perf_counter()
            - start_time
        )

        processing_fps = (

            frame_no / elapsed

            if elapsed > 0

            else 0.0
        )

        # ====================================================
        # INFORMATION OVERLAY
        # ====================================================

        cv2.putText(

            annotated,

            "YOLO-WORLD | OPEN VOCABULARY",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (0, 255, 0),

            2
        )

        cv2.putText(

            annotated,

            f"Objects: {object_count}",

            (20, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.70,

            (0, 255, 255),

            2
        )

        cv2.putText(

            annotated,

            f"FPS: {processing_fps:.1f}",

            (20, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )

        if live_camera:

            cv2.putText(

                annotated,

                "LIVE CAMERA",

                (20, 140),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (0, 255, 0),

                2
            )

        # ====================================================
        # SIDE-BY-SIDE DISPLAY
        # ====================================================

        if args.side_by_side:

            original = frame.copy()

            cv2.putText(

                original,

                "RGB CAMERA INPUT",

                (20, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.75,

                (0, 0, 255),

                2
            )

            view = cv2.hconcat(

                [
                    original,
                    annotated
                ]

            )

        else:

            view = annotated

        # ====================================================
        # SAVE
        # ====================================================

        writer.write(
            view
        )

        # ====================================================
        # LIVE DISPLAY
        # ====================================================

        cv2.imshow(

            "NMDC - YOLO World Object Detection",

            view
        )

        # ====================================================
        # PROGRESS FOR VIDEO
        # ====================================================

        if not live_camera:

            if frame_no % 15 == 0:

                if total > 0:

                    progress = (
                        frame_no
                        / total
                        * 100
                    )

                    print(

                        f"\rProgress: "
                        f"{progress:6.2f}% | "

                        f"Processing FPS: "
                        f"{processing_fps:5.1f}",

                        end=""
                    )

        else:

            if frame_no % 30 == 0:

                print(

                    f"\rLIVE | "
                    f"Frame: {frame_no} | "
                    f"Objects: {object_count} | "
                    f"FPS: {processing_fps:.1f}",

                    end=""
                )

        # ====================================================
        # KEYBOARD CONTROL
        # ====================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # Q or ESC
        if key == ord("q") or key == 27:

            print(
                "\nStopped by user."
            )

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")

    print("=" * 70)
    print("DETECTION COMPLETE")
    print("=" * 70)

    print(
        f"Frames processed : {frame_no}"
    )

    print(
        f"Saved to         : {args.output}"
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()