# Motion Storyboard Generator
# Run locally with: python -m streamlit run app.py

import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from pptx import Presentation
from pptx.util import Inches


FrameResult = Tuple[str, float, str]


OCR_LANGUAGES = {
    "English": "eng",
    "Korean": "kor",
    "Japanese": "jpn",
    "Spanish": "spa",
    "Portuguese": "por",
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Dutch": "nld",
    "Chinese Simplified": "chi_sim",
    "Chinese Traditional": "chi_tra",
    "Arabic": "ara",
    "Hindi": "hin",
    "English + Korean": "eng+kor",
    "English + Japanese": "eng+jpn",
    "English + Spanish": "eng+spa",
    "English + Portuguese": "eng+por",
    "English + French": "eng+fra",
    "English + German": "eng+deu",
    "English + Chinese Simplified": "eng+chi_sim",
}


def clean_text(text: str) -> str:
    """Clean OCR output so it is easier to compare frame-to-frame."""
    return " ".join(text.replace("\n", " ").split()).strip()


def detect_text(frame: np.ndarray, min_chars: int = 3, ocr_language: str = "eng") -> str:
    """
    Return OCR text from a frame, or an empty string if text is too short.

    This uses the stable OCR behavior, but allows different languages.
    If a selected language is not installed, it safely falls back to English.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Stable preprocessing.
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        raw_text = pytesseract.image_to_string(thresh, lang=ocr_language)
    except pytesseract.TesseractError:
        raw_text = pytesseract.image_to_string(thresh, lang="eng")

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


def text_similarity(a: str, b: str) -> float:
    """
    Compare detected OCR text between two frames.

    1.0 means the text is basically the same.
    Lower scores mean the copy changed.
    """
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def save_frame(frame: np.ndarray, output_dir: Path, index: int, timestamp: float) -> str:
    frame_filename = output_dir / f"frame_{index:03d}_{timestamp:.2f}s.jpg"
    cv2.imwrite(str(frame_filename), frame)
    return str(frame_filename)


def choose_best_frame_from_sequence(sequence: List[Dict]) -> Dict:
    """
    Pick one clean frame from a run of frames containing text.

    This keeps the deck clean by choosing the frame with the most complete OCR text.
    If there is a tie, it chooses the later frame.
    """
    return max(sequence, key=lambda item: (len(item["text"]), item["timestamp"]))


def extract_text_frames(
    video_path: str,
    sample_every_seconds: float = 0.25,
    min_chars: int = 3,
    duplicate_threshold: float = 8.0,
    sequence_gap_seconds: float = 0.25,
    ocr_language: str = "eng",
) -> List[FrameResult]:
    """
    Extract frames that contain text.

    Stable version:
    - scans for text
    - supports multiple OCR languages
    - groups nearby text frames together
    - keeps the most complete/later frame from each group
    - removes duplicates only when the image and OCR text are both similar
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
    last_saved_text = ""
    frame_index = 0

    def finish_sequence():
        nonlocal last_saved_frame, last_saved_text

        if not current_sequence:
            return

        best = choose_best_frame_from_sequence(current_sequence)

        is_duplicate = False
        if last_saved_frame is not None:
            diff_score = frame_difference(best["frame"], last_saved_frame)
            text_score = text_similarity(best["text"], last_saved_text)

            image_is_similar = diff_score < duplicate_threshold
            text_is_similar = text_score > 0.82

            # Only remove as duplicate if image AND text are similar.
            # This helps avoid skipping changed copy on the same background.
            is_duplicate = image_is_similar and text_is_similar

        if not is_duplicate:
            frame_path = save_frame(
                best["frame"],
                output_dir,
                len(text_frames) + 1,
                best["timestamp"],
            )
            text_frames.append((frame_path, best["timestamp"], best["text"]))
            last_saved_frame = best["frame"].copy()
            last_saved_text = best["text"]

        current_sequence.clear()

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            timestamp = frame_index / fps
            detected_text = detect_text(frame, min_chars=min_chars, ocr_language=ocr_language)

            if detected_text:
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


def create_full_ad_gif(
    video_path: str,
    output_dir: Path,
    max_width: int = 720,
    gif_fps: int = 8,
    max_duration_seconds: int = 45,
) -> str:
    """
    Create a GIF preview of the full ad for the final slide.

    This works better than embedded video when the deck is uploaded to Google Slides.
    """
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        fps = 30

    max_frames_to_read = min(total_frames, int(fps * max_duration_seconds))
    step = max(1, int(fps / gif_fps))

    frames = []
    frame_number = 0

    while frame_number < max_frames_to_read:
        success, frame = capture.read()
        if not success:
            break

        if frame_number % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            if width > max_width:
                scale = max_width / width
                new_size = (max_width, int(height * scale))
                rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(rgb))

        frame_number += 1

    capture.release()

    gif_path = output_dir / "full_motion_reference.gif"

    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / gif_fps),
            loop=0,
            optimize=True,
        )

    return str(gif_path)


def create_powerpoint(text_frames: List[FrameResult], video_path: str) -> str:
    """
    Create a PowerPoint storyboard deck.

    Layout:
    - One captured frame per slide
    - Captured frame on the left
    - Completely blank notes area on the right
    - Final slide includes a full-ad GIF reference instead of embedded video
    """
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    for i, (frame_path, timestamp, detected_text) in enumerate(text_frames, start=1):
        slide = prs.slides.add_slide(blank_layout)

        title_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.45))
        title_box.text_frame.text = f"Motion Storyboard QC — Frame {i}"

        timestamp_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.72), Inches(5.8), Inches(0.3))
        timestamp_box.text_frame.text = f"Timestamp: {timestamp:.2f}s"

        # Large captured frame on the left half of the slide.
        slide.shapes.add_picture(
            frame_path,
            Inches(0.6),
            Inches(1.15),
            width=Inches(5.9),
            height=Inches(4.45),
        )

        # Right side is intentionally blank for editors/reviewers to add notes, copy, or tables.
        notes_label = slide.shapes.add_textbox(Inches(6.85), Inches(0.72), Inches(5.85), Inches(0.3))
        notes_label.text_frame.text = "Notes / Copy / QC"

        blank_notes_box = slide.shapes.add_textbox(Inches(6.85), Inches(1.15), Inches(5.85), Inches(5.9))
        blank_notes_box.text_frame.text = ""
        blank_notes_box.line.width = 1

    # Final slide: full ad GIF reference.
    final_slide = prs.slides.add_slide(blank_layout)
    title_box = final_slide.shapes.add_textbox(Inches(0.4), Inches(0.3), Inches(12.5), Inches(0.5))
    title_box.text_frame.text = "Full Motion Reference"

    note_box = final_slide.shapes.add_textbox(Inches(0.8), Inches(0.9), Inches(11.8), Inches(0.5))
    note_box.text_frame.text = "Full ad preview GIF for Google Slides compatibility."

    gif_dir = Path(tempfile.mkdtemp())
    full_gif_path = create_full_ad_gif(video_path, gif_dir)

    final_slide.shapes.add_picture(
        full_gif_path,
        Inches(1.3),
        Inches(1.55),
        width=Inches(10.7),
        height=Inches(5.45),
    )

    output_path = str(Path(tempfile.mkdtemp()) / "motion_storyboard_deck.pptx")
    prs.save(output_path)
    return output_path


st.set_page_config(page_title="Motion Storyboard QC", layout="wide")

st.title("Motion Storyboard QC Generator")
st.write("Upload a motion video. The tool finds frames with visible copy and turns them into a storyboard deck.")

uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "m4v", "avi"])

with st.sidebar:
    st.header("OCR Language")
    selected_language_label = st.selectbox(
        "Choose the main text language",
        list(OCR_LANGUAGES.keys()),
        index=0,
    )
    selected_language_code = OCR_LANGUAGES[selected_language_label]

    st.caption(
        "If the ad has English plus another language, choose a combined option like English + Korean."
    )

    st.header("QC Detection Settings")
    sample_rate = st.slider("Scan every X seconds", 0.10, 2.0, 0.25, 0.05)
    min_chars = st.slider("Minimum OCR characters", 1, 20, 3, 1)
    duplicate_threshold = st.slider("Duplicate filter strength", 1.0, 25.0, 8.0, 1.0)
    sequence_gap = st.slider("Moving text grouping window", 0.25, 2.0, 0.25, 0.25)

st.info(
    "Stable default: scan every 0.25s, minimum OCR characters 3, duplicate strength 8, moving text window 0.25. "
    "Each captured frame gets its own slide with blank space for editor notes. The final slide includes the full motion reference."
)

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("Generate Storyboard"):
        with st.spinner(f"Scanning video for text frames using {selected_language_label} OCR..."):
            frames = extract_text_frames(
                video_path,
                sample_every_seconds=sample_rate,
                min_chars=min_chars,
                duplicate_threshold=duplicate_threshold,
                sequence_gap_seconds=sequence_gap,
                ocr_language=selected_language_code,
            )

        st.success(f"Found {len(frames)} storyboard frames with text.")

        if frames:
            st.subheader("Preview")
            cols = st.columns(3)
            for i, (frame_path, timestamp, detected_text) in enumerate(frames):
                with cols[i % 3]:
                    st.image(frame_path, caption=f"{timestamp:.2f}s")
                    st.caption(detected_text[:200])

            with st.spinner("Building storyboard deck with individual review slides..."):
                pptx_path = create_powerpoint(frames, video_path)

            with open(pptx_path, "rb") as f:
                st.download_button(
                    "Download Storyboard Deck for Google Slides",
                    data=f,
                    file_name="motion_storyboard_deck.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
        else:
            st.warning("No text frames found. Try lowering the minimum OCR characters, scanning more frequently, or choosing a different OCR language.")
