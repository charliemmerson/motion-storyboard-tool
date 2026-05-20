# Motion Storyboard Generator MVP
# Run with: streamlit run app.py

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from pptx import Presentation
from pptx.util import Inches


def has_visible_text(frame: np.ndarray, min_chars: int = 3) -> Tuple[bool, str]:
	"""Return True if OCR finds enough text in a video frame."""
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Boost contrast for OCR
	gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	text = pytesseract.image_to_string(thresh).strip()
	cleaned = " ".join(text.split())

	return len(cleaned) >= min_chars, cleaned


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
	"""Estimate how visually different two frames are."""
	a_small = cv2.resize(a, (160, 90))
	b_small = cv2.resize(b, (160, 90))
	diff = cv2.absdiff(a_small, b_small)
	return float(np.mean(diff))


def extract_text_frames(
	video_path: str,
	sample_every_seconds: float = 0.5,
	min_chars: int = 3,
	duplicate_threshold: float = 8.0,
) -> List[Tuple[str, float, str]]:
	"""Extract frames that contain text and skip near-duplicates."""
	output_dir = Path(tempfile.mkdtemp())
	capture = cv2.VideoCapture(video_path)

	fps = capture.get(cv2.CAP_PROP_FPS)
	if not fps or fps <= 0:
		fps = 30

	frame_interval = max(1, int(fps * sample_every_seconds))
	text_frames = []
	last_saved_frame = None
	frame_index = 0

	while True:
		success, frame = capture.read()
		if not success:
			break

		if frame_index % frame_interval == 0:
			contains_text, detected_text = has_visible_text(frame, min_chars=min_chars)

			if contains_text:
				is_duplicate = False
				if last_saved_frame is not None:
					diff_score = frame_difference(frame, last_saved_frame)
					is_duplicate = diff_score < duplicate_threshold

				if not is_duplicate:
					timestamp = frame_index / fps
					frame_filename = output_dir / f"frame_{len(text_frames) + 1:03d}_{timestamp:.2f}s.jpg"
					cv2.imwrite(str(frame_filename), frame)
					text_frames.append((str(frame_filename), timestamp, detected_text))
					last_saved_frame = frame.copy()

		frame_index += 1

	capture.release()
	return text_frames


def create_powerpoint(text_frames: List[Tuple[str, float, str]]) -> str:
	"""Create a PowerPoint storyboard with all extracted frames."""
	prs = Presentation()
	prs.slide_width = Inches(13.333)
	prs.slide_height = Inches(7.5)

	blank_layout = prs.slide_layouts[6]
	slide = prs.slides.add_slide(blank_layout)

	# Title
	title_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.4))
	title_box.text_frame.text = "Motion Storyboard QC"

	# Grid layout
	cols = 4
	rows = 2
	margin_x = 0.45
	margin_y = 0.85
	gap_x = 0.25
	gap_y = 0.35
	img_w = (13.333 - (2 * margin_x) - (cols - 1) * gap_x) / cols
	img_h = 2.55

	for i, (frame_path, timestamp, detected_text) in enumerate(text_frames[:8]):
		col = i % cols
		row = i // cols
		x = Inches(margin_x + col * (img_w + gap_x))
		y = Inches(margin_y + row * (img_h + gap_y + 0.35))

		slide.shapes.add_picture(frame_path, x, y, width=Inches(img_w), height=Inches(img_h))

		caption = slide.shapes.add_textbox(x, y + Inches(img_h + 0.05), Inches(img_w), Inches(0.3))
		caption.text_frame.text = f"{timestamp:.2f}s"

	# If more than 8 frames, create additional slides
	for start in range(8, len(text_frames), 8):
		slide = prs.slides.add_slide(blank_layout)
		title_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.4))
		title_box.text_frame.text = "Motion Storyboard QC continued"

		for j, (frame_path, timestamp, detected_text) in enumerate(text_frames[start:start + 8]):
			col = j % cols
			row = j // cols
			x = Inches(margin_x + col * (img_w + gap_x))
			y = Inches(margin_y + row * (img_h + gap_y + 0.35))
			slide.shapes.add_picture(frame_path, x, y, width=Inches(img_w), height=Inches(img_h))
			caption = slide.shapes.add_textbox(x, y + Inches(img_h + 0.05), Inches(img_w), Inches(0.3))
			caption.text_frame.text = f"{timestamp:.2f}s"

	output_path = str(Path(tempfile.mkdtemp()) / "motion_storyboard_qc.pptx")
	prs.save(output_path)
	return output_path


st.set_page_config(page_title="Motion Storyboard QC", layout="wide")

st.title("Motion Storyboard QC Generator")
st.write("Upload a motion video. The tool finds frames with visible copy and turns them into a storyboard deck.")

uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "m4v", "avi"])

with st.sidebar:
	st.header("Settings")
	sample_rate = st.slider("Scan every X seconds", 0.25, 2.0, 0.5, 0.25)
	min_chars = st.slider("Minimum OCR characters", 1, 20, 3, 1)
	duplicate_threshold = st.slider("Duplicate filter strength", 1.0, 25.0, 8.0, 1.0)

if uploaded_video:
	with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp:
		tmp.write(uploaded_video.read())
		video_path = tmp.name

	st.video(video_path)

	if st.button("Generate Storyboard"):
		with st.spinner("Scanning video for text frames..."):
			frames = extract_text_frames(
				video_path,
				sample_every_seconds=sample_rate,
				min_chars=min_chars,
				duplicate_threshold=duplicate_threshold,
			)

		st.success(f"Found {len(frames)} frames with text.")

		if frames:
			st.subheader("Preview")
			cols = st.columns(4)
			for i, (frame_path, timestamp, detected_text) in enumerate(frames):
				with cols[i % 4]:
					st.image(frame_path, caption=f"{timestamp:.2f}s")
					st.caption(detected_text[:120])

			pptx_path = create_powerpoint(frames)
			with open(pptx_path, "rb") as f:
				st.download_button(
					"Download PowerPoint Storyboard",
					data=f,
					file_name="motion_storyboard_qc.pptx",
					mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
				)
		else:
			st.warning("No text frames found. Try lowering the minimum OCR characters or scanning more frequently.")

