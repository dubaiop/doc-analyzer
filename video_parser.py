import base64
import io
import os
import tempfile

def extract_video_frames(content: bytes, num_frames: int = 6) -> list[dict]:
    """
    Extract evenly-spaced frames from a video file.
    Returns list of {"b64": str, "time": "M:SS"}.
    """
    import cv2
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    frames = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24

        if total <= 0:
            cap.release()
            return frames

        indices = [int(i * total / num_frames) for i in range(num_frames)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((640, 640))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            secs = idx / fps
            time_label = f"{int(secs // 60)}:{int(secs % 60):02d}"
            frames.append({"b64": b64, "time": time_label})

        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames
