# RunnerFinder – Number Search

RunnerFinder is a Python + Streamlit app that scans marathon photos and finds runners by their **number (bib)**.  
Given a folder of race photos and a target number, the app:

1. Runs OCR on every image.
2. Extracts all digit strings.
3. Computes how similar each detected number is to the target (using a fuzzy **LCS match ratio**).
4. Highlights all matches in the images and shows them in an interactive web interface.

---

## Screenshots

### Search example: runner 183

This screenshot shows RunnerFinder searching for runner number `183` with a minimum match ratio of `0.65`.  
Several images with numbers that partially match `183` (e.g., `103`, `1322`) are found and annotated.

![Search example for 183](docs/Screenshot1.png)

---

### Search example: runner 1437 (with progress bar)

While scanning the photo folder, the app shows the current image being processed, a progress bar, and the configured search settings in the sidebar.

![Search example for 1437 with progress bar](docs/Screenshot2.png)

---

### Search settings sidebar

The left sidebar lets the user:

- Enter the runner number they are looking for.
- Adjust the **minimum match ratio** (how strict matching should be for partially visible / occluded numbers).
- Start the search.

![Search settings sidebar](docs/Screenshot3.png)

---

### Result view: multiple matches in one image

Here the app finds both `1437` and `1438` in the same photo.  
For each detected number it shows:

- The detected digits.
- The **match ratio** relative to the requested number.
- The OCR model’s own confidence.

Bounding boxes are drawn around all matches.

![Result view with multiple numbers](docs/Screenshot4.png)

---

## Project structure

Current repository layout:

```text
RunnerFinder/
├── app.py             # Main Streamlit application
├── src/               # (Reserved for reusable modules / future refactoring)
├── photos/            # Input marathon photos (user-provided)
├── output/            # Optional: where annotated images can be saved (CLI version)
├── docs/              # Screenshots used in the README
├── requirements.txt   # Python dependencies
└── README.md
