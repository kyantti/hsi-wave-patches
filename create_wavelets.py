import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import shutil
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
INPUT_DIR = "data/raw/"  # Directory containing .npy files
OUTPUT_DIR = "data/interim/"  # Output directory
PATCH_SIZE = 32
SUB_PATCH_SIZE = 4

# Setup output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# Define wavelets - use a simple, robust approach
wavelets_cwt = ['morl']  # Morlet wavelet for CWT
wavelets_dwt = ['db1', 'db4', 'haar', 'db8']  # Common discrete wavelets

# ---------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------

def find_optimal_patch_position(hyperspectral_image):
    """
    Find the optimal position for patch extraction
    Uses the same logic as the reference code:
    - Center vertically (middle of image height)
    - Search horizontally from left to right for first valid position
    - Valid position = entire 32x32 patch contains no background (zero) pixels
    
    Args:
        hyperspectral_image: 3D numpy array (height, width, bands)
        
    Returns:
        tuple: (start_row, start_col) or None if no suitable position found
    """
    height, width, bands = hyperspectral_image.shape
    
    # Check if image is large enough
    if height < PATCH_SIZE or width < PATCH_SIZE:
        return None
    
    # Center the patch vertically (align middle of square with middle of image height)
    center_row = height // 2
    start_row = max(0, center_row - PATCH_SIZE // 2)  # Ensure the square fits within the image
    
    # Search horizontally from left to right for the first valid position
    for col in range(width - PATCH_SIZE + 1):
        # Extract the 32x32 patch at this position
        patch = hyperspectral_image[start_row:start_row + PATCH_SIZE, col:col + PATCH_SIZE, :]
        
        # Check if the entire patch contains no background pixels (non-zero pixels)
        # We'll check if all pixels in the patch are non-zero across all bands
        if np.all(patch != 0):
            print(f"Found valid patch position at row={start_row}, col={col}")
            return (start_row, col)
    
    # If no suitable position found, fall back to center position
    print("No suitable non-background position found, using center position")
    start_col = (width - PATCH_SIZE) // 2
    print(f"Using fallback position at row={start_row}, col={start_col}")
    return (start_row, start_col)

def create_rgb_image(hyperspectral_image):
    """
    Create an RGB image from hyperspectral data for visualization
    
    Args:
        hyperspectral_image: 3D numpy array (height, width, bands)
        
    Returns:
        RGB image as numpy array
    """
    height, width, bands = hyperspectral_image.shape
    
    # Select bands for RGB (adjust these indices based on your hyperspectral data)
    # Common approach: use bands that correspond roughly to red, green, blue wavelengths
    if bands >= 3:
        # For typical hyperspectral data, try to select bands around:
        # Red: ~670nm, Green: ~550nm, Blue: ~450nm
        # Here we'll use a simple approach of taking bands from different parts of the spectrum
        red_band = min(int(bands * 0.8), bands - 1)    # Later bands (longer wavelengths)
        green_band = min(int(bands * 0.5), bands - 1)  # Middle bands
        blue_band = min(int(bands * 0.2), bands - 1)   # Earlier bands (shorter wavelengths)
        
        rgb_image = np.stack([
            hyperspectral_image[:, :, red_band],
            hyperspectral_image[:, :, green_band],
            hyperspectral_image[:, :, blue_band]
        ], axis=2)
    else:
        # If less than 3 bands, use grayscale repeated 3 times
        gray_band = hyperspectral_image[:, :, 0] if bands > 0 else np.zeros((height, width))
        rgb_image = np.stack([gray_band, gray_band, gray_band], axis=2)
    
    # Normalize to 0-255 range
    rgb_image = rgb_image.astype(np.float64)
    for i in range(3):
        channel = rgb_image[:, :, i]
        channel_min, channel_max = channel.min(), channel.max()
        if channel_max > channel_min:
            rgb_image[:, :, i] = 255 * (channel - channel_min) / (channel_max - channel_min)
        else:
            rgb_image[:, :, i] = 0
    
    return rgb_image.astype(np.uint8)

def visualize_patches_on_image(hyperspectral_image, start_row, start_col, image_name, output_subdir):
    """
    Create a visualization showing the patch locations on the RGB image
    
    Args:
        hyperspectral_image: 3D numpy array (height, width, bands)
        start_row, start_col: Starting position of the main patch
        image_name: Name of the original image
        output_subdir: Output subdirectory
    """
    try:
        # Create RGB image
        rgb_image = create_rgb_image(hyperspectral_image)
        
        # Create a copy for drawing patches
        patch_visualization = rgb_image.copy()
        
        # Draw the main 32x32 patch boundary in red
        cv2_available = False
        try:
            import cv2
            cv2_available = True
        except ImportError:
            pass
        
        if cv2_available:
            import cv2
            # Draw main patch boundary
            cv2.rectangle(patch_visualization, 
                         (start_col, start_row), 
                         (start_col + PATCH_SIZE, start_row + PATCH_SIZE), 
                         (255, 0, 0), 2)
            
            # Draw 4x4 sub-patch grid
            for i in range(0, PATCH_SIZE, SUB_PATCH_SIZE):
                for j in range(0, PATCH_SIZE, SUB_PATCH_SIZE):
                    top_left = (start_col + j, start_row + i)
                    bottom_right = (start_col + j + SUB_PATCH_SIZE, start_row + i + SUB_PATCH_SIZE)
                    cv2.rectangle(patch_visualization, top_left, bottom_right, (0, 255, 0), 1)
        else:
            # Fallback using matplotlib for drawing rectangles
            import matplotlib.patches as patches
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original RGB image
            ax1.imshow(rgb_image)
            ax1.set_title('Original RGB Image')
            ax1.axis('off')
            
            # Image with patch visualization
            ax2.imshow(rgb_image)
            
            # Draw main patch boundary
            main_patch = patches.Rectangle((start_col, start_row), PATCH_SIZE, PATCH_SIZE,
                                         linewidth=2, edgecolor='red', facecolor='none')
            ax2.add_patch(main_patch)
            
            # Draw 4x4 sub-patch grid
            for i in range(0, PATCH_SIZE, SUB_PATCH_SIZE):
                for j in range(0, PATCH_SIZE, SUB_PATCH_SIZE):
                    sub_patch = patches.Rectangle((start_col + j, start_row + i), 
                                                SUB_PATCH_SIZE, SUB_PATCH_SIZE,
                                                linewidth=1, edgecolor='green', facecolor='none')
                    ax2.add_patch(sub_patch)
            
            ax2.set_title('Patch Locations (Red: Main 32x32, Green: 4x4 Sub-patches)')
            ax2.axis('off')
            
            # Save the visualization
            patch_viz_filename = f'patch_visualization_{image_name}.png'
            plt.savefig(os.path.join(output_subdir, patch_viz_filename), 
                       bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()
            return
        
        # If cv2 is available, create matplotlib visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original RGB image
        ax1.imshow(rgb_image)
        ax1.set_title('Original RGB Image')
        ax1.axis('off')
        
        # Image with patch overlays
        ax2.imshow(patch_visualization)
        ax2.set_title('Patch Locations (Red: Main 32x32, Green: 4x4 Sub-patches)')
        ax2.axis('off')
        
        # Save the visualization
        patch_viz_filename = f'patch_visualization_{image_name}.png'
        plt.savefig(os.path.join(output_subdir, patch_viz_filename), 
                   bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Error creating patch visualization: {e}")

def extract_patch_spectra(hyperspectral_image, start_row, start_col):
    """
    Extract 4x4 sub-patches from the 32x32 patch and compute mean spectra
    
    Args:
        hyperspectral_image: 3D numpy array (height, width, bands)
        start_row, start_col: Starting position of the 32x32 patch
        
    Returns:
        list: Mean spectra for each 4x4 sub-patch
    """
    patch_spectra = []
    
    # Extract 8x8 grid of 4x4 sub-patches
    for i in range(0, PATCH_SIZE, SUB_PATCH_SIZE):
        for j in range(0, PATCH_SIZE, SUB_PATCH_SIZE):
            # Extract 4x4 sub-patch
            sub_patch = hyperspectral_image[
                start_row + i:start_row + i + SUB_PATCH_SIZE,
                start_col + j:start_col + j + SUB_PATCH_SIZE,
                :
            ]
            
            # Compute mean spectrum across spatial dimensions
            mean_spectrum = np.mean(sub_patch, axis=(0, 1))
            patch_spectra.append(mean_spectrum)
    
    return patch_spectra

def generate_cwt_images(spectrum, patch_idx, image_name, output_subdir, class_folder):
    """
    Generate CWT images for a given spectrum
    
    Args:
        spectrum: 1D array representing the mean spectrum
        patch_idx: Index of the patch
        image_name: Name of the original image
        output_subdir: Output subdirectory for detailed structure
        class_folder: The class folder (e.g., 'C0') for simplified output
    """
    scales = np.arange(1, min(256, len(spectrum)))  # Adjust scales based on spectrum length
    
    for wvt in wavelets_cwt:
        try:
            if wvt == 'morl':
                coefficients, frequencies = pywt.cwt(spectrum, scales, wvt)
            elif wvt.startswith('shan'):
                wvt_param = 'shan0.5-1.0'
                coefficients, frequencies = pywt.cwt(spectrum, scales, wvt_param)
            else:
                continue
            
            coefficients_magnitude = np.abs(coefficients)
            wavelet_dir = os.path.join(output_subdir, f'CWT_{wvt}')
            os.makedirs(wavelet_dir, exist_ok=True)
            
            plt.figure(figsize=(2.24, 2.24))
            plt.imshow(coefficients_magnitude, cmap='viridis', aspect='auto')
            plt.axis('off')
            
            filename = f'CWT_{wvt}_{image_name}_patch_{patch_idx:02d}.png'
            plt.savefig(os.path.join(wavelet_dir, filename), 
                       bbox_inches='tight', pad_inches=0, dpi=100)
            
            # Also save to the simplified class directory, organized by wavelet type
            if class_folder:
                class_output_dir = os.path.join(OUTPUT_DIR, 'by-class', f'CWT_{wvt}', class_folder)
                os.makedirs(class_output_dir, exist_ok=True)
                plt.savefig(os.path.join(class_output_dir, filename),
                            bbox_inches='tight', pad_inches=0, dpi=100)

            plt.close()
            
        except Exception as e:
            print(f"Error processing wavelet {wvt} for patch {patch_idx}: {e}")
            continue

def generate_dwt_images(spectrum, patch_idx, image_name, output_subdir, class_folder):
    """
    Generate DWT images for a given spectrum
    
    Args:
        spectrum: 1D array representing the mean spectrum
        patch_idx: Index of the patch
        image_name: Name of the original image
        output_subdir: Output subdirectory for detailed structure
        class_folder: The class folder (e.g., 'C0') for simplified output
    """
    for wvt in wavelets_dwt:
        try:
            coeffs = pywt.dwt(spectrum, wvt)
            A, D = coeffs
            
            wavelet_dir = os.path.join(output_subdir, f'DWT_{wvt}')
            os.makedirs(wavelet_dir, exist_ok=True)
            
            plt.figure(figsize=(1.5, 1.5))
            plt.subplot(2, 1, 1)
            plt.plot(A)
            plt.axis('off')
            plt.subplot(2, 1, 2)
            plt.plot(D)
            plt.axis('off')
            
            filename = f'DWT_{wvt}_{image_name}_patch_{patch_idx:02d}.png'
            plt.savefig(os.path.join(wavelet_dir, filename), 
                       bbox_inches='tight', pad_inches=0, dpi=100)

            # Also save to the simplified class directory, organized by wavelet type
            if class_folder:
                class_output_dir = os.path.join(OUTPUT_DIR, 'by-class', f'DWT_{wvt}', class_folder)
                os.makedirs(class_output_dir, exist_ok=True)
                plt.savefig(os.path.join(class_output_dir, filename),
                            bbox_inches='tight', pad_inches=0, dpi=100)

            plt.close()
            
        except Exception as e:
            print(f"Error processing DWT {wvt} for patch {patch_idx}: {e}")
            continue

def process_hyperspectral_image(npy_file_path):
    """
    Process a single hyperspectral image (sequential processing only)
    
    Args:
        npy_file_path: Path to the .npy hyperspectral image
    """
    try:
        # Load hyperspectral image
        hyperspectral_image = np.load(npy_file_path)
        
        # Ensure correct dimensions (height, width, bands)
        if len(hyperspectral_image.shape) != 3:
            print(f"Invalid shape for {npy_file_path}: {hyperspectral_image.shape}")
            return
        
        print(f"Processing {npy_file_path} with shape: {hyperspectral_image.shape}")
        
        # Find optimal patch position
        patch_position = find_optimal_patch_position(hyperspectral_image)
        if patch_position is None:
            print(f"No suitable patch position found for {npy_file_path}")
            return
        
        start_row, start_col = patch_position
        
        # Extract class folder and image name from the path
        # Path structure: INPUT_DIR/CLASS_FOLDER/image_name.npy
        path_parts = os.path.normpath(npy_file_path).split(os.sep)
        class_folder = None
        image_name = os.path.splitext(os.path.basename(npy_file_path))[0]
        
        # Find the class folder (should be one of C0, C1, C2, C3, etc.)
        for part in path_parts:
            if part.startswith('C') and part[1:].isdigit():
                class_folder = part
                break
        
        # Create output subdirectory with class structure
        if class_folder:
            output_subdir = os.path.join(OUTPUT_DIR, class_folder, image_name)
        else:
            output_subdir = os.path.join(OUTPUT_DIR, image_name)
        
        os.makedirs(output_subdir, exist_ok=True)
        
        # Create patch visualization
        print(f"Creating patch visualization for {image_name} in class {class_folder}")
        visualize_patches_on_image(hyperspectral_image, start_row, start_col, image_name, output_subdir)
        
        # Extract mean spectra from 4x4 sub-patches
        patch_spectra = extract_patch_spectra(hyperspectral_image, start_row, start_col)
        
        # Process each patch spectrum sequentially
        for patch_idx, spectrum in enumerate(patch_spectra):
            print(f"Processing patch {patch_idx + 1}/64 for {image_name}")
            
            # Generate wavelet transform images
            generate_cwt_images(spectrum, patch_idx, image_name, output_subdir, class_folder)
            # Uncomment the line below if you also want DWT images
            # generate_dwt_images(spectrum, patch_idx, image_name, output_subdir, class_folder)
        
        print(f"Completed processing {npy_file_path}")
        
    except Exception as e:
        print(f"Error processing {npy_file_path}: {e}")

def process_single_image(npy_file):
    """
    Process a single image (wrapper function for backward compatibility)
    """
    npy_file_path = os.path.join(INPUT_DIR, npy_file)
    process_hyperspectral_image(npy_file_path)

def process_class_folder(class_folder):
    """
    Process all .npy files in a class folder (sequential processing only)
    
    Args:
        class_folder: Name of the class folder (e.g., 'C0', 'C1', etc.)
    """
    class_path = os.path.join(INPUT_DIR, class_folder)
    npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in {class_folder}")
        return
    
    print(f"Found {len(npy_files)} .npy files in {class_folder}")
    
    # Process files sequentially
    for npy_file in npy_files:
        npy_file_path = os.path.join(class_path, npy_file)
        process_hyperspectral_image(npy_file_path)

def process_all_data(use_parallel=False, max_workers=None):
    """
    Process all hyperspectral data with optional parallel processing
    
    Args:
        use_parallel (bool): Whether to use parallel processing
        max_workers (int): Maximum number of workers for parallel processing
    """
    # Find all class folders (C0, C1, C2, C3)
    class_folders = [f for f in os.listdir(INPUT_DIR) 
                     if os.path.isdir(os.path.join(INPUT_DIR, f)) and f.startswith('C')]
    
    if not class_folders:
        print("No class folders found in the input directory.")
        return
    
    print(f"Found class folders: {class_folders}")
    
    if use_parallel:
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)
        print(f"Using parallel processing with {max_workers} maximum workers")
        
        # Collect all file paths from all class folders
        all_file_paths = []
        for class_folder in class_folders:
            class_path = os.path.join(INPUT_DIR, class_folder)
            npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
            file_paths = [os.path.join(class_path, npy_file) for npy_file in npy_files]
            all_file_paths.extend(file_paths)
        
        print(f"Processing {len(all_file_paths)} files in parallel...")
        
        # Process all files in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_hyperspectral_image, all_file_paths)
    else:
        print("Using sequential processing")
        
        # Process each class folder sequentially
        for class_folder in class_folders:
            print(f"\n=== Processing class {class_folder} ===")
            process_class_folder(class_folder)
    
    print("\nProcessing completed!")

# ---------------------------------------------------------
# 3. Main processing
# ---------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process hyperspectral data with wavelet transforms')
    parser.add_argument('--parallel', action='store_true', 
                       help='Use parallel processing (default: sequential)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Process the data
    process_all_data(use_parallel=args.parallel, max_workers=args.workers)