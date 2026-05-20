# Motion Storyboard Generator MVP
# Run with: streamlit run app.py

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytesseract
import streamlit as st
from pptx import Presentation
from pptx.util import Inches


FrameResult = Tuple[str, float, str]


def clean_text(text: str) -> str:
    """Clean OCR output so it is easier to compare frame-to-frame."""
    return " ".join(text.replace("\n", " ").split()).strip()


def detect_text(frame: np.ndarray, min_chars: int = 3) -> str:
    """Return OCR text from a frame, or an empty string if text is too short."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Upscale and sharpen a bit for better OCR results.
    gray = cv2.resize(gray, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    raw_text = pytesseract.image_to_string(thresh)
    cleaned = clean_text(raw_text)

    if len(cleaned) < min_chars:
        return ""

    return cleaned


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Estimate how visually different two frames are."""
    a_small = cv2.resize(a, (160, 90))
    b_small = cv2.resize(b, (160, 90))
    diff = cv2.absdiff(a_small, b_small)
    return float(np.mean(diff))


def choose_best_frame_from_text_sequence(sequence: List[Dict]) -> Dict:
    """
    Pick the best frame from a run of frames containing text.

    This fixes moving/scrolling text better than saving the first text frame.
    It prefers the frame with the most complete OCR text. If there is a tie,
    it picks the later frame, which usually captures the final sentence state.
    """
    return max(
        sequence,
        key=lambda item: (len(item["text"]), item["timestamp"]),
    )


def save_frame(frame: np.ndarray, output_dir: Path, index: int, timestamp: float) -> str:
    frame_filename = output_dir / f"frame_{index:03d}_{timestamp:.2f}s.jpg"
    cv2.imwrite(str(frame_filename), frame)
    return str(frame_filename)


def extract_text_frames(
    video_path: str,
    sample_every_seconds: float = 0.25,
    min_chars: int = 3,
    duplicate_threshold: float = 8.0,
    sequence_gap_seconds: float = 0.75,
) -> List[FrameResult]:
    """
    Extract frames that contain text.

    Important upgrade:
    Instead of saving the FIRST frame where text appears, this groups nearby
    text frames together and saves the most complete/later frame from each group.
    This helps with scrolling or animated sentences.
    """
    output_dir = Path(tempfile.mkdtemp())
    capture = cv2.VideoCapture(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    frame_interval = max(1, int(fps * sample_every_seconds))
    max_gap_frames = max(1, int(fps * sequence_gap_seconds))

    text_frames: List[FrameResult] = []
    current_sequence: List[Dict] = []
    last_text_frame_index = None
    last_saved_frame = None
    frame_index = 0

    def finish_sequence():
        nonlocal last_saved_frame
        if not current_sequence:
            return

        best = choose_best_frame_from_text_sequence(current_sequence)

        is_duplicate = False
        if last_saved_frame is not None:
            diff_score = frame_difference(best["frame"], last_saved_frame)
            is_duplicate = diff_score < duplicate_threshold

        if not is_duplicate:
            frame_path = save_frame(
                best["frame"],
                output_dir,
                len(text_frames) + 1,
                best["timestamp"],
            )
            text_frames.append((frame_path, best["timestamp"], best["text"]))
            last_saved_frame = best["frame"].copy()

        current_sequence.clear()

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            timestamp = frame_index / fps
            detected_text = detect_text(frame, min_chars=min_chars)

            if detected_text:
                # If this text frame is far away from the last text frame,
                # close the previous sequence and start a new one.
                if (
                    last_text_frame_index is not None
                    and frame_index - last_text_frame_index > max_gap_frames
                ):
                    finish_sequence()

                current_sequence.append(
                    {
                        "frame": frame.copy(),
                        "timestamp": timestamp,
                        "text": detected_text,
                    }
                )
                last_text_frame_index = frame_index

            else:
                # If text disappeared for long enough, close the sequence.
                if (
                    current_sequence
                    and last_text_frame_index is not None
                    and frame_index - last_text_frame_index > max_gap_frames
                ):
                    finish_sequence()

        frame_index += 1

    finish_sequence()
    capture.release()
    return text_frames


def create_powerpoint(text_frames: List[FrameResult]) -> str:
    """Create a PowerPoint storyboard with extracted frames and OCR text."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    cols = 3
    rows = 2
    frames_per_slide = cols * rows
    margin_x = 0.45
    margin_y = 0.85
    gap_x = 0.25
    gap_y = 0.35
    img_w = (13.333 - (2 * margin_x) - (cols - 1) * gap_x) / cols
    img_h = 2.25

    for slide_start in range(0, len(text_frames), frames_per_slide):
        slide = prs.slides.add_slide(blank_layout)
        title_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.4))
        title_box.text_frame.text = "Motion Storyboard QC"

        for j, (frame_path, timestamp, detected_text) in enumerate(
            text_frames[slide_start:slide_start + frames_per_slide]
        ):
            col = j % cols
            row = j // cols
            x = Inches(margin_x + col * (img_w + gap_x))
            y = Inches(margin_y + row * (img_h + gap_y + 0.55))

            slide.shapes.add_picture(frame_path, x, y, width=Inches(img_w), height=Inches(img_h))

            caption = slide.shapes.add_textbox(x, y + Inches(img_h + 0.04), Inches(img_w), Inches(0.45))
            caption.text_frame.text = f"{timestamp:.2f}s | {detected_text[:90]}"

    output_path = str(Path(tempfile.mkdtemp()) / "motion_storyboard_qc.pptx")
    prs.save(output_path)
    return output_path


st.set_page_config(page_title="Motion Storyboard QC", layout="wide")

st.title("Motion Storyboard QC Generator")
st.write("Upload a motion video. The tool finds frames with visible copy and turns them into a storyboard deck.")

uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "m4v", "avi"])

with st.sidebar:
    st.header("Settings")
    sample_rate = st.slider("Scan every X seconds", 0.25, 2.0, 0.25, 0.25)
    min_chars = st.slider("Minimum OCR characters", 1, 20, 3, 1)
    duplicate_threshold = st.slider("Duplicate filter strength", 1.0, 25.0, 8.0, 1.0)
    sequence_gap = st.slider("Moving text grouping window", 0.25, 2.0, 0.75, 0.25)

st.info(
    "Tip: For moving or scrolling text, the app now groups nearby text frames and saves the most complete/later frame instead of grabbing the first text frame."
)

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("Generate Storyboard"):
        with st.spinner("Scanning video for final/complete text frames..."):
            frames = extract_text_frames(
                video_path,
                sample_every_seconds=sample_rate,
                min_chars=min_chars,
                duplicate_threshold=duplicate_threshold,
                sequence_gap_seconds=sequence_gap,
            )

        st.success(f"Found {len(frames)} storyboard frames with text.")

        if frames:
            st.subheader("Preview")
            cols = st.columns(3)
            for i, (frame_path, timestamp, detected_text) in enumerate(frames):
                with cols[i % 3]:
                    st.image(frame_path, caption=f"{timestamp:.2f}s")
                    st.caption(detected_text[:200])

            pptx_path = create_powerpoint(frames)
            with open(pptx_path, "rb") as f:
                st.download_button(
                    "Download Storyboard Deck for Google Slides",
                    data=f,
                    file_name="motion_storyboard_deck.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
        else:
            st.warning("No text frames found. Try lowering the minimum OCR characters or scanning more frequently.")

