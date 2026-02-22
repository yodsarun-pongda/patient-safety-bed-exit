from __future__ import annotations

import time
import threading
from dataclasses import asdict
from typing import Any, Dict, Generator, Optional, Union

import cv2
from model.PostureStatus import PostureStatus
from model.StreamConfig import StreamConfig

try:
    from starlette.responses import StreamingResponse
except Exception:  # pragma: no cover
    StreamingResponse = None  # type: ignore

# Key point
from model.PoseConfig import map_keypoints_xy_conf, classify_posture_coco17

class PostureDetectionService:
    def __init__(
        self,
        source: Union[int, str] = 0,
        config: Optional[StreamConfig] = None,
    ):
        self.source = source
        self.config = config or StreamConfig()

        self.status = PostureStatus()

        self._cap: Optional[cv2.VideoCapture] = None
        self._source_mode: str = "camera"
        self._video_files: list[str] = []
        self._video_index: int = 0
        self._current_video_path: Optional[str] = None
        self._lock = threading.Lock()
        self._stopped = False

        self._min_frame_interval = 1.0 / max(float(self.config.target_fps), 1e-6)

        # We load YOLO only when it is actually needed.
        self._yolo = None

        # Keep latest inference result so skipped frames still look smooth.
        self._frame_idx = 0
        self._last_annotated = None

        # Simple FPS counter for overlay text.
        self._fps_last_t = time.time()
        self._fps_count = 0
        self._fps_value = 0.0

        # Error throttle: same message won't be printed every frame.
        self._last_err_msg: Optional[str] = None
        self._last_err_print_t: float = 0.0

        # Last inference metadata (so skipped frames still have correct status)
        self._last_person_count: int = 0
        self._last_posture: str = self.status.posture
        self._last_det_score: float = float(self.status.det_score)
        self._last_pose_score: float = float(self.status.pose_score)
        self._last_infer_t: float = 0.0
        self._active_streams: int = 0
        self._last_jpg: Optional[bytes] = None
        self._next_stream_id: int = 0
        self._producer_stream_id: Optional[int] = None

        print(f"Posture Detection Service have been initialized")

    def _is_camera_stream(self) -> bool:
        return bool(getattr(self.config, "isCameraStream", True))

    def _get_local_video_source(self) -> str:
        from pathlib import Path

        candidates: list[str] = []
        local_source = getattr(self.config, "local_video_source", None)
        if isinstance(local_source, str) and local_source.strip():
            candidates.append(local_source.strip())
        if isinstance(self.source, str) and self.source.strip():
            candidates.append(self.source.strip())
        if not candidates:
            candidates.append("mock-video")

        for candidate in candidates:
            resolved = Path(candidate)
            if resolved.exists():
                return str(resolved)

            app_relative = Path(__file__).resolve().parents[1] / candidate
            if app_relative.exists():
                return str(app_relative)

        return candidates[0]

    # Open stream
    def open(self, *, force_reopen: bool = False) -> None:
        with self._lock:
            # New stream session should always clear stopped state.
            self._stopped = False

            if (not force_reopen) and self._cap is not None and self._cap.isOpened():
                self._update_status(message="stream_opened")
                return

            # Clean stale handle before reopening.
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

            capture_source: Union[int, str] = self.source
            if self._is_camera_stream():
                self._source_mode = "camera"
            else:
                from pathlib import Path

                local_source = self._get_local_video_source()
                capture_source = local_source

                if Path(local_source).is_dir():
                    ok = self.load_local_video(local_source)
                    if not ok:
                        raise RuntimeError(
                            f"Cannot open local video directory {local_source!r}. "
                            "Check local_video_source path and video files."
                        )
                    self._update_status(message="stream_opened")
                    return

                self._source_mode = "video_file"

            # Camera backends sometimes need a short retry window after disconnect.
            last_error = ""
            for _ in range(5):
                cap = cv2.VideoCapture(capture_source)
                if cap is not None and cap.isOpened():
                    self._cap = cap
                    break
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                last_error = f"Cannot open source {capture_source!r}"
                time.sleep(0.2)

            if self._cap is None:
                raise RuntimeError(
                    f"{last_error}. "
                    "Check camera index / file path / RTSP URL."
                )
            self._update_status(message="stream_opened")

    # Try Close stream
    def close(self) -> None:
        with self._lock:
            self._stopped = True
            self._active_streams = 0
            self._producer_stream_id = None
            if self._cap is not None:
                try:
                    self._cap.release()
                finally:
                    self._cap = None
            self._update_status(
                message="stopped",
                posture="unknown",
                person_count=0,
                det_score=0.0,
                pose_score=0.0,
            )

    # Stop and closed client connection
    def stop(self) -> None:
        self.close()

    # Update current status
    def _update_status(
        self,
        *,
        message: Optional[str] = None,
        posture: Optional[str] = None,
        person_count: Optional[int] = None,
        det_score: Optional[float] = None,
        pose_score: Optional[float] = None,
        alert_message: Optional[str] = None,
    ) -> None:
        if message is not None:
            self.status.message = str(message)
        if posture is not None:
            self.status.posture = str(posture)
        if person_count is not None:
            self.status.person_count = int(person_count)
        if det_score is not None:
            self.status.det_score = float(det_score)
        if pose_score is not None:
            self.status.pose_score = float(pose_score)
        if alert_message is not None:
            self.status.alert_message = str(alert_message)
            self.status.alert_seq = int(self.status.alert_seq) + 1
        self.status.updated_at = time.time()

    # Getting frame from video
    def _read_frame(self):
        cap = self._cap
        if cap is None:
            return False, None

        ok, frame = cap.read()
        if ok:
            return True, frame

        # For a directory of demo videos, go to the next file and loop forever.
        if self._source_mode == "video_dir":
            files = getattr(self, "_video_files", None) or []
            if not files:
                return False, None
            next_index = (int(getattr(self, "_video_index", 0)) + 1) % len(files)
            if self._open_video_file_by_index(next_index):
                next_cap = self._cap
                if next_cap is None:
                    return False, None
                return next_cap.read()
            return False, None

        # For a single local video file, optionally restart from frame 0.
        if self._source_mode == "video_file" and self.config.loop_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok2, frame2 = cap.read()
            return ok2, frame2

        return False, None

    # Use to resizing the image
    # def _maybe_resize(self, frame):
    #     if frame is None:
    #         return None
    #     w, h = self.config.resize_width, self.config.resize_height
    #     if w is None and h is None:
    #         return frame
    #     if w is None:
    #         scale = h / frame.shape[0]
    #         w = int(frame.shape[1] * scale)
    #     if h is None:
    #         scale = w / frame.shape[1]
    #         h = int(frame.shape[0] * scale)
    #     return cv2.resize(frame, (int(w), int(h)))

    # Encoding image for response to api
    def _encode_jpeg(self, frame) -> Optional[bytes]:
        if frame is None:
            return None
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)]
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            return None
        return buf.tobytes()

    # Chunking the image before response
    def _mjpeg_chunk(self, jpg: bytes) -> bytes:
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )

    # Input text to image
    def _draw_text(self, frame, text: str, org=(10, 25)):
        x, y = org
        # Draw rectangle with text
        cv2.rectangle(frame, (x - 6, y - 18), (x + 560, y + 12), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _validate_is_yolo(self) -> None:
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "ultralytics is not installed. Install it with: pip install ultralytics"
            ) from e
        self._yolo = YOLO(self.config.yolo_pose_model)
    
    # Count total person in frame
    def _inquiry_and_count_person(self, r0):
        try:
            boxes = getattr(r0, "boxes", None)
            if boxes is None:
                return 0
            return len(boxes)
        except Exception:
            return 0

    # Detect pose by skeleton
    def _acquire_pose(self, frame_bgr):
        self._validate_is_yolo()
        if self._yolo is None:
            return frame_bgr, 0, "unknown", 0.0, 0.0

        # Capturing/Getting frame
        results = self._yolo(frame_bgr, conf=float(self.config.conf), verbose=False)
        if not results:
            return frame_bgr, 0, "unknown", 0.0, 0.0

        # Getting frame
        r0 = results[0]

        # Counting person 
        person_count = self._inquiry_and_count_person(r0)
        posture = "unknown"
        det_score = 0.0
        pose_score = 0.0

        # Use Ultralytics built-in drawing for skeleton/keypoints.
        annotated = frame_bgr

        # Draw skeleton to image
        if self.config.draw_skeleton and person_count <= 1:
            annotated = r0.plot(boxes=False)

        boxes = getattr(r0, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None:
            confs = boxes.conf.cpu().numpy()
            if len(confs) > 0:
                det_score = float(max(confs))

        kps = getattr(r0, "keypoints", None)
        if kps is not None and getattr(kps, "xy", None) is not None:
            xy_all = kps.xy.cpu().numpy()
            cf_all = kps.conf.cpu().numpy() if kps.conf is not None else None
            if len(xy_all) > 0:
                xy = xy_all[0]
                cf = cf_all[0] if cf_all is not None and len(cf_all) > 0 else None
                keypoints_by_name = map_keypoints_xy_conf(xy, cf)
                posture = classify_posture_coco17(keypoints_by_name)

                if cf is not None and len(cf) > 0:
                    valid_conf = [float(c) for c in cf if float(c) > 0.0]
                    if valid_conf:
                        pose_score = float(sum(valid_conf) / len(valid_conf))

        if person_count <= 0:
            posture = "unknown"
            pose_score = 0.0
        elif person_count > 1:
            posture = "Multi Person detected"
            pose_score = 0.0

        if self.config.draw_bbox:
            self.draw_label(
                r0=r0,
                text=posture,
                annotated=annotated,
                isMultiplePerson=person_count > 1,
            )

        return annotated, person_count, posture, det_score, pose_score
    
    def draw_label(self, r0, text, annotated, isMultiplePerson: bool):
        try:
            # Decision  rectangle color
            color_code = (0, 255, 0)
            if isMultiplePerson:
                color_code = (0, 0, 255)
                # Overall warning text (top-left)
                cv2.putText(
                    annotated,
                    "MULTIPLE PERSON",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            # Get meta data from each freme
            boxes = getattr(r0, "boxes", None)
            if boxes is not None and boxes.xyxy is not None and boxes.conf is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color_code, 2)
                    
                    # Label text depends on whether multiple people are detected
                    label = "Multiple person" if isMultiplePerson else text

                    # Increase font size and thickness
                    font_scale = 1
                    thickness = 2

                    # Get text size for background box
                    (tw, th), baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        thickness,
                    )

                    text_x = x1
                    text_y = max(th + 10, y1 - 10)

                    # Draw filled background rectangle (black)
                    cv2.rectangle(
                        annotated,
                        (text_x - 5, text_y - th - 8),
                        (text_x + tw + 5, text_y + 5),
                        (0, 0, 0),
                        -1,
                    )

                    # Draw text on top (red)
                    cv2.putText(
                        annotated,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color_code,
                        thickness,
                        cv2.LINE_AA,
                    )
        except Exception:
            pass

    # def get_key_point_skeleton(self, r0, annotated, person_count):
    #     kps = getattr(r0, "keypoints", None)
    #     if kps is None:
    #         return annotated, person_count

    #     xy = kps.xy.cpu().numpy()

    #     conf = kps.conf.cpu().numpy() if kps.conf is not None else None

    #     print("xy shape:", xy.shape)
    #     print("conf shape:", None if conf is None else conf.shape)

    def _open_video_file_by_index(self, index: int) -> bool:
        import cv2

        files = getattr(self, "_video_files", None)
        if not files:
            print("[_open_video_file_by_index] No video file list loaded")
            self._cap = None
            return False

        if index < 0 or index >= len(files):
            print(f"[_open_video_file_by_index] Invalid index: {index}")
            self._cap = None
            return False

        old_cap = getattr(self, "_cap", None)
        if old_cap is not None:
            try:
                old_cap.release()
            except Exception:
                pass

        path = files[index]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[_open_video_file_by_index] Cannot open video: {path}")
            self._cap = None
            return False

        self._cap = cap
        self._video_index = index
        self._current_video_path = path
        self._source_mode = "video_dir"

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(
                f"[_open_video_file_by_index] Opened ({index+1}/{len(files)}): {path} | "
                f"{w}x{h} | fps={fps:.2f} | frames={total}"
            )
        except Exception:
            pass

        return True

    def load_local_video(self, video_dir: str = "mock-video") -> bool:
        from pathlib import Path

        if not video_dir or not isinstance(video_dir, str):
            print("[load_local_video_dir] Invalid directory path")
            return False

        p = Path(video_dir)
        if not p.exists() or not p.is_dir():
            print(f"[load_local_video_dir] Directory not found: {video_dir}")
            return False

        exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".mpeg", ".mpg"}
        files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts])

        if not files:
            print(f"[load_local_video_dir] No video files found in: {video_dir}")
            return False

        # release old capture
        old_cap = getattr(self, "_cap", None)
        if old_cap is not None:
            try:
                old_cap.release()
            except Exception:
                pass

        self._video_files = [str(f) for f in files]
        self._video_index = 0

        return self._open_video_file_by_index(self._video_index)

    # --------------------------
    # Streaming
    # --------------------------
    def mjpeg_frames(self) -> Generator[bytes, None, None]:
        with self._lock:
            self._active_streams += 1
            self._next_stream_id += 1
            stream_id = self._next_stream_id
            if self._producer_stream_id is None:
                self._producer_stream_id = stream_id

        last_time = 0.0

        try:
            while True:
                with self._lock:
                    if self._stopped:
                        break
                    if self._producer_stream_id is None:
                        self._producer_stream_id = stream_id
                    is_producer = self._producer_stream_id == stream_id

                # Only one producer stream runs camera + inference.
                # Other clients reuse the latest JPEG to avoid camera contention/freezes.
                if not is_producer:
                    if self._last_jpg is not None:
                        yield self._mjpeg_chunk(self._last_jpg)
                    time.sleep(max(0.01, self._min_frame_interval))
                    continue

                # Keep output frame rate close to target_fps.
                now = time.time()
                sleep_for = self._min_frame_interval - (now - last_time)
                if sleep_for > 0:
                    time.sleep(sleep_for)

                # Optional: intentionally slow down the loop (milliseconds).
                # You can set this in StreamConfig as `extra_sleep_ms`.
                extra_sleep_ms = float(getattr(self.config, "extra_sleep_ms", 0) or 0)
                if extra_sleep_ms > 0:
                    time.sleep(extra_sleep_ms / 1000.0)

                last_time = time.time()

                # Ensure camera handle is alive for this iteration.
                cap_ok = self._cap is not None and self._cap.isOpened()
                if not cap_ok:
                    try:
                        self.open(force_reopen=self._cap is not None)
                    except Exception as e:
                        self._update_status(
                            message=f"open_error: {type(e).__name__}",
                            posture="unknown",
                            person_count=0,
                            det_score=0.0,
                            pose_score=0.0,
                        )
                        if self._last_jpg is not None:
                            yield self._mjpeg_chunk(self._last_jpg)
                        time.sleep(0.25)
                        continue

                # Read frame from camera or local video based on config.
                ok, frame = self._read_frame()
                
                if not ok:
                    # Try to reopen once when client reconnects or camera backend drops.
                    try:
                        self.open(force_reopen=True)
                        ok, frame = self._read_frame()
                    except Exception:
                        ok, frame = False, None

                
                if not ok:
                    self._update_status(
                        message="waiting_frame",
                        posture="unknown",
                        person_count=0,
                        det_score=0.0,
                        pose_score=0.0,
                    )
                    if self._last_jpg is not None:
                        yield self._mjpeg_chunk(self._last_jpg)
                    time.sleep(0.1)
                    continue
                
                # Resizing image for improve performance
                # frame = self._maybe_resize(frame)
                if frame is None:
                    self._update_status(
                        message="invalid_frame",
                        posture="unknown",
                        person_count=0,
                        det_score=0.0,
                        pose_score=0.0,
                    )
                    if self._last_jpg is not None:
                        yield self._mjpeg_chunk(self._last_jpg)
                    continue

                # Optional - Flip image for
                # frame = cv2.flip(frame, 1)

                self._frame_idx += 1

                annotated = frame

                # --------------------------
                # Frame skipping (process every N frames) + optional time-based throttle
                # --------------------------
                every_n = max(1, int(getattr(self.config, "infer_every_n_frames", self.config.infer_every_n_frames)))
                run_infer = (self._frame_idx % every_n) == 0

                # Optional: also enforce a minimum time between inferences (milliseconds).
                # Set StreamConfig.infer_min_interval_ms (e.g., 300 = run YOLO at most ~3 FPS)
                infer_min_interval_ms = int(getattr(self.config, "infer_min_interval_ms", 0) or 0)
                if infer_min_interval_ms > 0:
                    now_t = time.time()
                    if (now_t - self._last_infer_t) * 1000.0 < float(infer_min_interval_ms):
                        run_infer = False

                # Detect skeleton & draw edge
                person_count = 0
                posture = self._last_posture
                det_score = float(self._last_det_score)
                pose_score = float(self._last_pose_score)
                if run_infer or self._last_annotated is None:
                    try:
                        annotated, person_count, posture, det_score, pose_score = self._acquire_pose(frame)
                        self._last_annotated = annotated
                        self._last_person_count = int(person_count)
                        self._last_posture = str(posture)
                        self._last_det_score = float(det_score)
                        self._last_pose_score = float(pose_score)
                        self._last_infer_t = time.time()
                        self._update_status(
                            message="running",
                            posture=posture,
                            person_count=person_count,
                            det_score=det_score,
                            pose_score=pose_score,
                        )
                    except Exception as e:
                        # Handle failed
                        annotated = frame
                        msg = f"{type(e).__name__}: {str(e)}".strip()
                        show = msg if len(msg) <= 90 else (msg[:87] + "...")
                        # self._draw_text(annotated, f"YOLO err: {show}")

                        # keep previous count when inference fails
                        person_count = int(self._last_person_count)
                        posture = self._last_posture
                        det_score = float(self._last_det_score)
                        pose_score = float(self._last_pose_score)

                        t = time.time()
                        if msg != self._last_err_msg or (t - self._last_err_print_t) > 3.0:
                            print("[YOLO ERROR]", msg)
                            self._last_err_msg = msg
                            self._last_err_print_t = t

                        self._update_status(
                            message=f"yolo_error: {show}",
                            posture=posture,
                            person_count=person_count,
                            det_score=det_score,
                            pose_score=pose_score,
                        )
                else:
                    # Skip inference: reuse last annotated + last count
                    annotated = self._last_annotated if self._last_annotated is not None else frame
                    person_count = int(self._last_person_count)
                    posture = self._last_posture
                    det_score = float(self._last_det_score)
                    pose_score = float(self._last_pose_score)
                    self._update_status(
                        message="running",
                        posture=posture,
                        person_count=person_count,
                        det_score=det_score,
                        pose_score=pose_score,
                    )

                # Use for place information
                self.draw_information(text=self._last_posture, person_count=person_count, annotated=annotated)

                jpg = self._encode_jpeg(annotated)
                if jpg is None:
                    if self._last_jpg is not None:
                        yield self._mjpeg_chunk(self._last_jpg)
                    continue

                self._last_jpg = jpg
                yield self._mjpeg_chunk(jpg)
        except Exception as e:
            print(f"Error while process post detection: {str(e)}")
        finally:
            remaining_streams = 0
            with self._lock:
                self._active_streams = max(0, self._active_streams - 1)
                if self._producer_stream_id == stream_id:
                    self._producer_stream_id = None
                remaining_streams = self._active_streams

            # Keep camera open even when no clients are connected.
            # Camera is closed only via explicit close()/stop().
            if remaining_streams == 0 and not self._stopped:
                self._update_status(
                    message="idle_camera_open",
                    posture=self._last_posture,
                    person_count=self._last_person_count,
                    det_score=self._last_det_score,
                    pose_score=self._last_pose_score,
                )

    def draw_information(self, text, person_count, annotated):
        # Show basic detection status.
        if self.config.draw_status_text:
            status = "SIT"
            if person_count > 1:
                status = f"MULTIPLE PERSONS {person_count}"
            elif person_count == 0:
                status = "NO PERSON"
            else:
                status = text
            self._draw_text(annotated, f"{status} | count={person_count} | conf>={self.config.conf:.2f}")

        # Update and draw FPS once per second.
        if self.config.draw_fps:
            self._fps_count += 1
            dt = time.time() - self._fps_last_t
            if dt >= 1.0:
                self._fps_value = self._fps_count / dt
                self._fps_count = 0
                self._fps_last_t = time.time()
            self._draw_text(annotated, f"FPS: {self._fps_value:.1f}", org=(10, 55))

    def get_status(self) -> Dict[str, Any]:
        data = asdict(self.status)
        data["updated_at"] = int(float(data.get("updated_at", time.time())) * 1000)
        return data

    def mjpeg_response(self):
        if StreamingResponse is None:
            raise RuntimeError(
                "StreamingResponse is not available. Install fastapi/starlette: pip install fastapi"
            )
        return StreamingResponse(
            self.mjpeg_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

cfg = StreamConfig(
    resize_width=640,
    yolo_pose_model="yolov8n-pose.pt",
    conf=0.35,
    infer_every_n_frames=1,
    target_fps=15,
    jpeg_quality=80,
)

postureDetectionService = PostureDetectionService(source=0, config=cfg)
