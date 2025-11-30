import argparse
from pathlib import Path

import cv2
import easyocr


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search marathon photos for a given bib number."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="photos",
        help="Folder with marathon images.",
    )
    parser.add_argument(
        "--bib",
        type=str,
        required=True,
        help="Bib number to search for, e.g. 17342.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Folder where annotated matches will be saved.",
    )
    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.6,
        help="Min fraction of bib digits that must match (for partial/occluded bibs).",
    )
    return parser.parse_args()


def load_images(images_dir: Path):
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in exts:
            yield path


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    bib = normalize_digits(args.bib)
    min_ratio = args.min_match_ratio

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not bib:
        raise ValueError("Bib must contain at least one digit.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading images from: {images_dir.resolve()}")
    print(f"🎯 Searching for bib number: {bib}")
    print(f"📏 Min match ratio: {min_ratio}")
    print(f"💾 Saving matched images to: {output_dir.resolve()}")

    print("🔎 Initializing OCR model (this may take a bit the first time)...")
    reader = easyocr.Reader(["en"])

    matches = []

    for img_path in load_images(images_dir):
        print(f"\n[INFO] Processing: {img_path.name}")
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        results = reader.readtext(img)  # [ [bbox, text, confidence], ... ]

        all_numbers = []
        for (bbox, text, conf) in results:
            digits = normalize_digits(text)
            if digits:
                all_numbers.append((digits, bbox, conf))

        print("  All detected number-like strings:")
        for digits, _, conf in all_numbers:
            print(f"   - {digits} (conf={conf:.2f})")

        found_in_this_image = False

        # Draw boxes for ALL matches (including partial bibs)
        for digits, bbox, conf in all_numbers:
            if is_bib_match(digits, bib, min_match_ratio=min_ratio):
                found_in_this_image = True
                print(f"  ✅ Match: OCR='{digits}' (conf={conf:.2f})")

                # bbox is 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                pts = [(int(x), int(y)) for (x, y) in bbox]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{digits}",
                    (x_min, max(y_min - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        if found_in_this_image:
            matches.append(img_path)
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), img)
            print(f"  💾 Saved annotated image: {out_path}")
        else:
            print("  ❌ No matching bib here.")

    print("\n===== SUMMARY =====")
    if matches:
        print(f"Found bib {bib} in {len(matches)} image(s):")
        for m in matches:
            print(f"  - {m}")
    else:
        print(f"No images found containing bib {bib}.")


if __name__ == "__main__":
    main()
