import streamlit as st
from pathlib import Path
import cv2
import easyocr


# ---------- Helper functions ----------

def normalize_digits(text: str) -> str:
    """Keep only digit characters from a string."""
    return "".join(ch for ch in text if ch.isdigit())


def lcs_length(a: str, b: str) -> int:
    """Length of the longest common subsequence between a and b."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def match_ratio(candidate: str, target: str) -> float:
    """
    Return the LCS-based match ratio between candidate and target.
    Ratio = LCS(candidate, target) / len(target).

    Both strings are first converted to digits-only.
    """
    candidate = normalize_digits(candidate)
    target = normalize_digits(target)
    if not candidate or not target:
        return 0.0

    lcs = lcs_length(candidate, target)
    return lcs / len(target)


def is_number_match(candidate: str, target: str, min_match_ratio: float = 0.6) -> bool:
    """Return True if the match ratio is at least min_match_ratio."""
    return match_ratio(candidate, target) >= min_match_ratio


@st.cache_resource
def load_reader():
    """Cache the OCR model so it is loaded only once."""
    return easyocr.Reader(["en"])


def process_image(img_path: Path, number: str, reader, min_match_ratio: float):
    """
    Run OCR on one image, draw boxes around matches, and
    return the annotated RGB image plus a list of matches.

    Each match dict contains:
        - digits: detected digits
        - conf: OCR confidence
        - bbox: bounding box
        - match_ratio: LCS-based match ratio to the target number
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, []

    results = reader.readtext(img)  # [ [bbox, text, conf], ... ]
    matches = []

    for bbox, text, conf in results:
        digits = normalize_digits(text)
        if not digits:
            continue

        mr = match_ratio(digits, number)  # between 0 and 1

        if mr >= min_match_ratio:
            matches.append(
                {"digits": digits, "conf": conf, "bbox": bbox, "match_ratio": mr}
            )

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

number_input = st.sidebar.text_input("Enter the runner number", "")
min_ratio = st.sidebar.slider(
    "Minimum match ratio (for partial / occluded numbers)",
    min_value=0.4,
    max_value=1.0,
    value=0.6,
    step=0.05,
)

start_search = st.sidebar.button("Search")

st.write(f"Using images from: `{images_dir.resolve()}`")

if start_search:
    number_digits = normalize_digits(number_input)
    if not number_digits:
        st.warning("Please enter a number that contains at least one digit.")
        st.stop()

    st.markdown(f"### Looking for number: `{number_digits}`")
    st.write("Running OCR on images. The first run may take some time.")

    reader = load_reader()

    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in exts
    )

    if not image_paths:
        st.warning("No images found in the photos folder.")
        st.stop()

    progress = st.progress(0)
    status_text = st.empty()

    total_matches = 0
    num_images = len(image_paths)

    for idx, img_path in enumerate(image_paths, start=1):
        status_text.text(f"Processing image {idx} of {num_images}: {img_path.name}")

        img_rgb, matches = process_image(
            img_path, number_digits, reader, min_match_ratio=min_ratio
        )

        if matches:
            total_matches += 1
            st.subheader(img_path.name)

            # Show both match ratio and OCR confidence
            match_strings = [
                f"{m['digits']} (match={m['match_ratio']:.2f}, ocr_conf={m['conf']:.2f})"
                for m in matches
            ]
            st.write("Matches in this image: " + ", ".join(match_strings))
            st.image(img_rgb, use_container_width=True)

        progress.progress(idx / num_images)

    status_text.empty()

    if total_matches == 0:
        st.info(
            f"No images found containing a number similar to `{number_digits}` "
            f"(with match ratio ≥ {min_ratio:.2f})."
        )
    else:
        st.success(
            f"Found {total_matches} image(s) with a number similar to `{number_digits}` "
            f"(match ratio ≥ {min_ratio:.2f})."
        )
