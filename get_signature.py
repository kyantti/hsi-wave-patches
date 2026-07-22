import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
FILE_PATH = "data/raw/C3/Fx10_20230717_Riego_C3_2023-07-17_09-16-20_ann0.npy"
PATCH_SIZE = 32
SUB_PATCH_SIZE = 4
TARGET_SUB_PATCH_INDEX = 0  # Corresponds to '_patch_00'

# ---------------------------------------------------------
# 2. Helper function to find the patch
# ---------------------------------------------------------

def find_optimal_patch_position(hyperspectral_image):
    """
    Find the optimal position for patch extraction based on the logic
    from create_wavelets.py.
    """
    height, width, _ = hyperspectral_image.shape

    if height < PATCH_SIZE or width < PATCH_SIZE:
        print("Image is smaller than the patch size.")
        return None

    # Center the patch vertically
    center_row = height // 2
    start_row = max(0, center_row - PATCH_SIZE // 2)

    # Search horizontally for a valid position (no zero-pixels)
    for col in range(width - PATCH_SIZE + 1):
        patch = hyperspectral_image[start_row:start_row + PATCH_SIZE, col:col + PATCH_SIZE, :]
        if np.all(patch != 0):
            print(f"Found valid patch position at row={start_row}, col={col}")
            return (start_row, col)

    # Fallback to the center if no fully non-zero patch is found
    print("No suitable non-background position found, using center position as fallback.")
    start_col = (width - PATCH_SIZE) // 2
    return (start_row, start_col)

# ---------------------------------------------------------
# 3. Main script execution
# ---------------------------------------------------------

def main():
    """
    Main function to load data, extract signature, and plot it.
    """
    # Check if the file exists
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    # 1. Load the hyperspectral image
    print(f"Loading image: {FILE_PATH}")
    hyperspectral_image = np.load(FILE_PATH)
    print(f"Image loaded with shape: {hyperspectral_image.shape}")

    # 2. Find the 32x32 patch position
    position = find_optimal_patch_position(hyperspectral_image)
    if position is None:
        return
    start_row, start_col = position

    # 3. Extract the main 32x32 patch
    main_patch = hyperspectral_image[start_row:start_row + PATCH_SIZE, start_row:start_row + PATCH_SIZE, :]

    # 4. Identify and extract the target 4x4 sub-patch
    # The sub-patches are indexed row by row.
    num_sub_patches_per_dim = PATCH_SIZE // SUB_PATCH_SIZE
    sub_patch_row = TARGET_SUB_PATCH_INDEX // num_sub_patches_per_dim
    sub_patch_col = TARGET_SUB_PATCH_INDEX % num_sub_patches_per_dim

    sub_start_row = sub_patch_row * SUB_PATCH_SIZE
    sub_start_col = sub_patch_col * SUB_PATCH_SIZE

    sub_patch = main_patch[sub_start_row:sub_start_row + SUB_PATCH_SIZE,
                           sub_start_col:sub_start_col + SUB_PATCH_SIZE, :]

    print(f"Extracted 4x4 sub-patch of shape: {sub_patch.shape}")

    # 5. Calculate the average hyperspectral signature
    # We average the values across the 4x4 spatial dimension for each band.
    average_signature = np.mean(sub_patch, axis=(0, 1))
    print(f"Calculated average signature of shape: {average_signature.shape}")

    # 6. Plot the signature
    plt.figure(figsize=(12, 6))
    plt.plot(average_signature)
    plt.title(f"Hyperspectral Signature of 4x4 region from\n{os.path.basename(FILE_PATH)}")
    plt.xlabel("Band Number")
    plt.ylabel("Average Reflectance")
    plt.grid(True)

    # 7. Save the plot
    output_filename = "hyperspectral_signature_C3.png"
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
