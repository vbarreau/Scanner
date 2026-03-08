# CT Scanner

A Python toolkit for processing DICOM CT scan data: load DICOM series, build 3D images, visualize slices, generate projections, export STL meshes, and render rotation videos.

## Features

- Load a folder of DICOM files into a 3D NumPy volume
- Interactive GUI (Tkinter) to browse slices, adjust contrast and compression
- Pseudo-radiography projections along X, Y, Z axes
- 360° rotation video export via GPU volume rendering (Vispy)
- STL mesh export via marching cubes (scikit-image + trimesh)

## Project structure

```
src/
  app.py          — Tkinter GUI application (entry point)
  Image.py        — Image3D class: load, process, export, visualize
  tranche.py      — Tranche class: wraps a single DICOM slice (pydicom)
  func.py         — Pure functions: compress, contrast, apply_gradient
  Radio.py        — Radio class: display a single 2D DICOM file
  tri_images.py   — Utilities for comparing and sorting DICOM series
Tests/
  test_func.py    — Tests for func.py (no DICOM files required)
  test_Image.py   — Tests for Image3D factories (no DICOM files required)
  test_tranches.py
requirements.txt
```

## Installation

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### GUI

```bash
python src/app.py
```

1. Click **Browse…** and select a folder containing DICOM files (`.dcm` or no extension).
2. The scan loads automatically. Scan properties are shown in the left panel.
3. Choose an action:
   - **Edit Image** — navigate slices (X/Y/Z sliders), apply contrast and compression.
   - **Create Animation** — configure rotation axis, camera POV, frame count, then export an `.mp4`.
   - **Create pseudo radiography** — view and save X/Y/Z projections as PNG.

### Scripting

```python
from src.Image import Image3D

# Load from a folder of DICOM files
scan = Image3D("path/to/dicom/folder")

# Basic processing
scan.change_contrast(30)        # -100 to 100
scan.compress(seuil=200, fac=0.5)
scan.seuil(100)                 # zero out values below threshold

# Inspect
print(scan.properties())        # shape, min, max, mean, zeros

# Export
scan.to_step("output.stl")
scan.rotation_video("output.mp4", n_frames=72, axis='z', pov='x')
```

## Running tests

```bash
python -m pytest Tests/
```

Tests in `test_func.py` and `test_Image.py` run without any DICOM data.

## Dependencies

| Package | Purpose |
|---|---|
| `pydicom` | Read DICOM files |
| `numpy` | Array operations |
| `scipy` | Morphological filtering |
| `scikit-image` | Marching cubes surface extraction |
| `trimesh` | STL mesh export |
| `matplotlib` | Slice visualization, histogram |
| `vispy` + `PyOpenGL` | GPU volume rendering for animations |
| `opencv-python` | Video encoding |
| `tqdm` | Progress bars |
