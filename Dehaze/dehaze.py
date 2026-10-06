import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def clahe_enhance(frame, clip_limit=2.0, grid_size=(8, 8)):
    """CIMFR-style HSV + CLAHE enhancement."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(grid_size)
    )
    v_enh = clahe.apply(v)

    enhanced = cv2.cvtColor(
        cv2.merge((h, s, v_enh)),
        cv2.COLOR_HSV2BGR
    )
    return enhanced


def simplest_color_balance(frame, percent=1.0):
    """Percentile color balancing, vectorized for real-time use."""
    if percent <= 0 or percent >= 100:
        return frame

    half = percent / 200.0
    output_channels = []

    for channel in cv2.split(frame):
        flat = np.sort(channel.reshape(-1))
        n = flat.size

        low_idx = max(0, int(n * half))
        high_idx = min(n - 1, int(n * (1.0 - half)))

        low = float(flat[low_idx])
        high = float(flat[high_idx])

        channel = np.clip(channel, low, high).astype(np.float32)
        channel = cv2.normalize(
            channel, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        output_channels.append(channel)

    return cv2.merge(output_channels)


def estimate_visibility_proxy(frame):
    """
    Image-only visibility proxy.
    This is NOT a calibrated physical visibility measurement in metres.
    Higher values generally indicate stronger visible local structure.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contrast = float(np.std(gray))

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    score = 0.7 * contrast + 300.0 * edge_density
    return score


def adaptive_enhance(frame):
    """
    Adaptive enhancement:
    - Always apply moderate CLAHE.
    - Apply stronger color balance when the image has low structure.
    """
    proxy = estimate_visibility_proxy(frame)

    base = clahe_enhance(
        frame,
        clip_limit=2.0,
        grid_size=(8, 8)
    )

    # Thresholds are prototype values, not mine-certified limits.
    if proxy < 65:
        base = clahe_enhance(
            base,
            clip_limit=3.0,
            grid_size=(8, 8)
        )
        base = simplest_color_balance(base, percent=1.0)
    elif proxy < 90:
        base = simplest_color_balance(base, percent=0.7)

    return base, proxy


def open_source(source):
    try:
        return int(source)
    except ValueError:
        return source


def main():
    parser = argparse.ArgumentParser(
        description="NMDC HEMM Task 1 - Fog Dehazing/Enhancement"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="0 for webcam, or path to a foggy video/image stream"
    )
    parser.add_argument(
        "--mode",
        choices=["clahe", "adaptive"],
        default="adaptive",
        help="enhancement mode"
    )
    parser.add_argument(
        "--output",
        default="outputs/enhanced_output.mp4",
        help="output video filename"
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="display original and enhanced frames together"
    )
    args = parser.parse_args()

    source = open_source(args.source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or fps_in <= 1:
        fps_in = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer_size = (width * 2, height) if args.side_by_side else (width, height)

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        writer_size
    )

    frame_count = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.mode == "clahe":
            enhanced = clahe_enhance(frame)
            proxy = estimate_visibility_proxy(frame)
        else:
            enhanced, proxy = adaptive_enhance(frame)

        elapsed = time.perf_counter() - t0
        fps_proc = (frame_count + 1) / elapsed if elapsed > 0 else 0.0

        display_enhanced = enhanced.copy()
        cv2.putText(
            display_enhanced,
            f"Mode: {args.mode}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            display_enhanced,
            f"Proxy visibility: {proxy:.1f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )
        cv2.putText(
            display_enhanced,
            f"Processing FPS: {fps_proc:.1f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if args.side_by_side:
            display_original = frame.copy()
            cv2.putText(
                display_original,
                "Original",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            view = np.hstack((display_original, display_enhanced))
        else:
            view = display_enhanced

        writer.write(view)
        cv2.imshow("NMDC HEMM - Fog Enhancement", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        frame_count += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Saved enhanced video to: {args.output}")
    print(f"Processed frames: {frame_count}")


if __name__ == "__main__":
    main()
