"""Stacked original-vs-segmentation comparison video for the Stage 1 (no-map) 360SFUDA output.

Consumes what video360_segformer.py already wrote - the per-frame overlay PNGs in frames/ and
the indoor/outdoor decision in indoor_outdoor_status.csv - and pairs each overlay with its own
source frame. The original panel is the raw frame resized to the model's input geometry rather
than the blurred tensor the overlay is drawn on, so the two panels share columns exactly and
the only difference between them is the segmentation itself.

The timeline strip needs the whole decision sequence, which is why this is a second pass over
the clip rather than something the inference script could draw as it went.

    python make_segmentation_comparison.py --video <src.mp4> --seg_dir <out> --output <cmp.mp4>
"""
import argparse
import csv
import os
import subprocess

import cv2
import numpy as np

PANEL_W, PANEL_H = 2048, 400          # the model's input geometry; overlays are saved at this size
STRIP_H = 56                          # indoor/outdoor timeline below both panels
BAR_TOP, BAR_H = 6, 26                # the coloured bar within the strip

OUTDOOR_BGR = (0, 255, 0)             # same green/red the border in video360_segformer.py uses
INDOOR_BGR = (0, 0, 255)
STRIP_BG = (28, 28, 28)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_status(seg_dir):
    """frame index -> is_indoor, as written by the inference pass."""
    path = os.path.join(seg_dir, "indoor_outdoor_status.csv")
    status = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            status[int(row["frame"])] = row["is_indoor"].strip().lower() == "true"
    if not status:
        raise RuntimeError(f"no rows in {path}")
    return status


def build_strip_base(status, n_frames):
    """The static part of the timeline: one column per horizontal pixel, coloured by that
    pixel's frame. Drawn once and copied per frame, since only the playhead moves."""
    strip = np.full((STRIP_H, PANEL_W, 3), STRIP_BG, dtype=np.uint8)
    for x in range(PANEL_W):
        frame = min(n_frames - 1, int(x * n_frames / PANEL_W))
        colour = INDOOR_BGR if status.get(frame, False) else OUTDOOR_BGR
        strip[BAR_TOP:BAR_TOP + BAR_H, x] = colour

    # Mark every transition so the boundary reads as a boundary and not a colour gradient.
    prev = status.get(0, False)
    for frame in range(1, n_frames):
        cur = status.get(frame, prev)
        if cur != prev:
            x = int(frame * PANEL_W / n_frames)
            cv2.line(strip, (x, BAR_TOP - 3), (x, BAR_TOP + BAR_H + 2), (255, 255, 255), 1)
        prev = cur

    ty = BAR_TOP + BAR_H + 18
    cv2.rectangle(strip, (8, ty - 10), (22, ty), INDOOR_BGR, -1)
    cv2.putText(strip, "indoor", (28, ty), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(strip, (100, ty - 10), (114, ty), OUTDOOR_BGR, -1)
    cv2.putText(strip, "outdoor", (120, ty), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return strip


def timeline_for_frame(strip_base, frame, n_frames, fps):
    strip = strip_base.copy()
    x = int(frame * PANEL_W / max(1, n_frames))
    x = min(PANEL_W - 1, x)
    cv2.line(strip, (x, 0), (x, BAR_TOP + BAR_H + 3), (255, 255, 255), 2)

    total = n_frames / fps
    cv2.putText(strip, f"0:00", (PANEL_W - 300, BAR_TOP + BAR_H + 18), FONT, 0.45,
                (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(strip, f"{int(total // 60)}:{total % 60:04.1f}",
                (PANEL_W - 110, BAR_TOP + BAR_H + 18), FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return strip


def caption(img, text, org, scale=0.7):
    """Text with a dark plate behind it, so it stays readable over sky or pale pavement."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def overlay_path(frames_dir, frame):
    return os.path.join(frames_dir, f"frame_{frame:06d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="the source 360 clip")
    ap.add_argument("--seg_dir", required=True, help="video360_segformer.py output directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=float, default=None, help="defaults to the source fps")
    ap.add_argument("--crf", type=int, default=20)
    args = ap.parse_args()

    frames_dir = os.path.join(args.seg_dir, "frames")
    status = load_status(args.seg_dir)
    n_overlays = len([f for f in os.listdir(frames_dir) if f.startswith("frame_")])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = args.fps or src_fps

    n_frames = min(n_overlays, src_frames, len(status))
    print(f"[INFO] source {src_frames} frames @ {src_fps:.3f} fps | overlays {n_overlays} | "
          f"status rows {len(status)} -> pairing {n_frames}")

    strip_base = build_strip_base(status, n_frames)
    out_h = PANEL_H * 2 + STRIP_H

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{PANEL_W}x{out_h}",
         "-r", f"{fps}", "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output],
        stdin=subprocess.PIPE)

    written = 0
    for frame in range(n_frames):
        ret, src_bgr = cap.read()
        if not ret:
            print(f"[WARN] source ended early at frame {frame}")
            break

        seg = cv2.imread(overlay_path(frames_dir, frame))
        if seg is None:
            print(f"[WARN] missing overlay for frame {frame}; stopping")
            break
        if seg.shape[:2] != (PANEL_H, PANEL_W):
            seg = cv2.resize(seg, (PANEL_W, PANEL_H))

        # Raw resize, not the smoothed tensor the overlay sits on: the left-hand term of the
        # comparison should be the video as shot.
        top = cv2.resize(src_bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
        caption(top, "ORIGINAL", (20, 40))

        t = frame / fps
        caption(seg, f"frame {frame:05d}   {int(t // 60)}:{t % 60:04.1f}",
                (20, PANEL_H - 20), scale=0.6)

        canvas = np.vstack([top, seg, timeline_for_frame(strip_base, frame, n_frames, fps)])
        ffmpeg.stdin.write(canvas.tobytes())
        written += 1
        if frame % 200 == 0:
            print(f"[INFO] {frame}/{n_frames}", flush=True)

    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {ffmpeg.returncode}")
    print(f"[DONE] {written} frames -> {args.output} ({PANEL_W}x{out_h} @ {fps:.3f} fps)")


if __name__ == "__main__":
    main()
