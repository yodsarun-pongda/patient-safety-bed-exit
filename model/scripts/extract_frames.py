import argparse, os, glob
import cv2

def list_videos(input_path: str):
    print(f"List file on {input_path}")
    if os.path.isdir(input_path):
        exts = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.MP4", "*.MOV")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(input_path, e)))
        
        print(f"Found {len(files)} videos")
        return sorted(files)
    return [input_path]

def main():
    # Getting ARGs
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Video file or folder containing videos")
    ap.add_argument("--output", required=True, help="Output folder for extracted frames")
    ap.add_argument("--fps", type=float, default=1.0, help="Target extraction fps (e.g., 1 means 1 frame/sec)")
    ap.add_argument("--max_frames", type=int, default=0, help="Optional cap. 0 = no cap")
    args = ap.parse_args()

    # Create output folder
    print(f"Output dir = {args.output}")
    os.makedirs(args.output, exist_ok=True)

    # List Video file
    videos = list_videos(args.input)
    if not videos:
        raise SystemExit("No videos found. Put videos in model/data/raw_videos/")

    # Extract frame from video files
    for vp in videos:

        # Use Open CV for extract Frame
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"[WARN] cannot open: {vp}")
            continue

        # Get FPS from source video or default 30
        source_video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Calculate Frame per sec
        step = max(int(round(source_video_fps / args.fps)), 1)

        # Build Path
        base = os.path.splitext(os.path.basename(vp))[0]
        out_dir = os.path.join(args.output, base)
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        idx = 0

        while True:
            # Read frame
            ok, frame = cap.read()
            if not ok:
                break
            
            # Select frame to getting
            if idx % step == 0:
                # Save frame to file
                out = os.path.join(out_dir, f"{base}_{idx:08d}.jpg")
                cv2.imwrite(out, frame)
                saved += 1

                # Limit total image
                if args.max_frames and saved >= args.max_frames:
                    break
            idx += 1

        # Close video
        cap.release()
        print(f"[OK] {base}: saved {saved} frames -> {out_dir}")

if __name__ == "__main__":
    main()
