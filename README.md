# Hyperspectral Image Wavelet Patch Processing

This project processes hyperspectral images by extracting patches and applying wavelet transforms to generate training data for machine learning models.

## Features

- Extracts 32x32 patches from hyperspectral images
- Divides patches into 64 sub-patches (4x4 each)
- Applies Continuous Wavelet Transform (CWT) to each sub-patch spectrum
- Generates visualization images showing patch locations on RGB representation
- Supports both sequential and parallel processing
- Centralized control over parallelization

## Requirements

This project uses `uv` for dependency management. The main dependencies are:
- numpy
- matplotlib
- pywt (PyWavelets)
- opencv-python (optional, for better patch visualization)

## Usage

### Quick Test (Single File)
```bash
uv run python process_single.py
```

### Test with Specific File
```bash
uv run python process_single.py path/to/your/file.npy
```

### Run System Test
```bash
uv run python test_processing.py
```

### Sequential Processing (Default)
```bash
uv run python main.py
```

### Parallel Processing
```bash
uv run python main.py --parallel
```

### Parallel Processing with Custom Worker Count
```bash
uv run python main.py --parallel --workers 4
```

### Help
```bash
uv run python main.py --help
```

## Command Line Options

- `--parallel`: Enable parallel processing (default: sequential)
- `--workers N`: Set number of parallel workers (default: auto-detect based on CPU cores)

## Input Structure

The script expects the following directory structure:

```
data/
└── cropped-hypercubes/
    ├── C0/
    │   ├── image1.npy
    │   ├── image2.npy
    │   └── ...
    ├── C1/
    │   ├── image1.npy
    │   └── ...
    └── ...
```

## Output Structure

The script generates the following output structure organized by class:

```
out/
├── C0/
│   ├── image1/
│   │   ├── patch_visualization_image1.png  # RGB visualization with patch locations
│   │   ├── CWT_morl/
│   │   │   ├── CWT_morl_image1_patch_00.png
│   │   │   ├── CWT_morl_image1_patch_01.png
│   │   │   └── ... (64 patches total)
│   │   └── DWT_db1/  # If DWT is enabled
│   │       ├── DWT_db1_image1_patch_00.png
│   │       └── ...
│   └── image2/
│       └── ...
├── C1/
│   ├── image1/
│   └── ...
├── C2/
│   └── ...
└── C3/
    └── ...
```

## Processing Details

1. **Patch Positioning**: 
   - **Vertical**: Center the 32x32 patch vertically (middle of image height)
   - **Horizontal**: Search from left to right for the first position where the entire 32x32 patch contains no background (zero) pixels
   - **Fallback**: If no valid position is found, use the center position
   
2. **Sub-patch Division**: The 32x32 patch is divided into 64 sub-patches of 4x4 pixels each (8x8 grid)

3. **Spectrum Calculation**: Mean spectrum is calculated for each 4x4 sub-patch across all spectral bands

4. **Wavelet Transform**: Continuous Wavelet Transform (Morlet wavelet) is applied to each spectrum to generate 2D images

5. **Visualization**: RGB representation of the original image with patch locations overlaid:
   - **Red rectangle**: Shows the 32x32 main patch boundary
   - **Green rectangles**: Show the 64 individual 4x4 sub-patches

## Parallel Processing

The script supports centralized parallel processing control:

- **Sequential Mode**: Processes files and patches one by one (default)
- **Parallel Mode**: Processes multiple files simultaneously using multiple CPU cores
- The parallelization is applied at the file level, not at the patch level, to maintain memory efficiency

## Configuration

Key parameters can be modified in the script:

- `PATCH_SIZE = 32`: Size of the main patch
- `SUB_PATCH_SIZE = 4`: Size of each sub-patch
- `INPUT_DIR`: Directory containing hyperspectral data
- `OUTPUT_DIR`: Directory for output files

## Band Selection for RGB Visualization

The script automatically selects bands for RGB visualization:
- Red channel: Band from ~80% of the spectrum (longer wavelengths)
- Green channel: Band from ~50% of the spectrum (middle wavelengths)  
- Blue channel: Band from ~20% of the spectrum (shorter wavelengths)

This provides a reasonable RGB representation for most hyperspectral datasets.
