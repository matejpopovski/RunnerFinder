import streamlit as st
from pathlib import Path
import cv2
import easyocr
import numpy as np


# ---------- Helper functions ----------

def normalize_digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def lcs_length(a: str, b: str) -> int:
    """Length of longest common subsequence between a and b."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def is_bib_match(candidate: str, target: str, min_match_ratio: float = 0.6) -> bool:
    """
    candidate: OCR digits (e.g. '280')
    target: true bib digits (e.g. '3280')
    """
    candidate = normalize_digits(candidate)
    target = normalize_digits(target)
    if not candidate or not target:
        return False

    lcs = lcs_length(candidate, target)
    ratio = lcs / len(target)
    return ratio >= min_match_ratio


@st.cache_resource
def load_reader():
    # cache the OCR model so it doesn't reload every time
    return easyocr.Reader(["en"])


def process_image(img_path: Path, bib: str, reader, min_match_ratio: float):
    """Run OCR on one image, return annotated RGB image + list of matches."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, []

    results = reader.readtext(img)  # [ [bbox, text, conf], ... ]

    matches = []
    for bbox, text, conf in results:
        digits = normalize_digits(text)
        if not digits:
            continue

        if is_bib_match(digits, bib, min_match_ratio=min_match_ratio):
            matches.append({"digits": digits, "conf": conf, "bbox": bbox})

            # draw box on image
            pts = [(int(x), int(y)) for (x, y) in bbox]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                img,
                digits,
                (x_min, max(y_min - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    # convert BGR -> RGB for Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, matches


# ---------- Streamlit UI ----------

st.title("RunnerFinder – Number Search")

images_dir = Path("photos")

if not images_dir.exists():
    st.error(f"Images folder not found: {images_dir.resolve()}")
    st.stop()

st.sidebar.header("Search Settings")

bib_input = st.sidebar.text_input("Enter your bib number", "")
min_ratio = st.sidebar.slider(
    "Min match ratio (for partial/occluded bibs)",
    min_value=0.4,
    max_value=1.0,
    value=0.6,
    step=0.05,
)

start_search = st.sidebar.button("Search")

st.write(f"Using images from: `{images_dir.resolve()}`")

if start_search:
    bib_digits = normalize_digits(bib_input)
    if not bib_digits:
        st.warning("Please enter a bib number that contains at least one digit.")
        st.stop()

    st.write(f"Looking for bib: **{bib_digits}**")
    st.write("Running OCR on images (first run can be slow)...")

    reader = load_reader()

    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in exts
    )

    if not image_paths:
        st.warning("No images found in the photos folder.")
        st.stop()

    total_matches = 0

    for img_path in image_paths:
        img_rgb, matches = process_image(
            img_path, bib_digits, reader, min_match_ratio=min_ratio
        )

        if matches:
            total_matches += 1
            st.subheader(img_path.name)

            match_strings = [
                f"{m['digits']} (conf={m['conf']:.2f})" for m in matches
            ]
            st.write("Matches in this image:", ", ".join(match_strings))
            st.image(img_rgb, use_column_width=True)

    if total_matches == 0:
        st.info(f"No images found containing bib similar to **{bib_digits}**.")
    else:
        st.success(f"Found {total_matches} image(s) with matching bib.")
