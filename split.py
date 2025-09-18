import os
import shutil
from sklearn.model_selection import train_test_split

# --- Configuration ---
INPUT_DIR = "data/interim/CWT_morl"    # The folder with your current class subdirectories (C0, C1, etc.)
OUTPUT_DIR = "data/processed"       # The new root folder for your train/test splits
TRAIN_RATIO = 0.8         # 80% for training, 20% for testing
RANDOM_STATE = 42         # Use a fixed seed for reproducible splits

def copy_files(file_list, dest_dir):
    """Copies a list of files to a destination directory."""
    for file_path in file_list:
        # The destination path is the new directory + the original filename
        dest_path = os.path.join(dest_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dest_path)

def main():
    """Main function to run the script."""
    # Check if the input directory exists
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return

    # Create the main output directories (train and test)
    train_dir = os.path.join(OUTPUT_DIR, "train")
    test_dir = os.path.join(OUTPUT_DIR, "test")

    if os.path.exists(OUTPUT_DIR):
        print(f"Warning: Output directory '{OUTPUT_DIR}' already exists. Files may be overwritten.")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get the list of class names (subdirectories in the input folder)
    class_names = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    print(f"Found {len(class_names)} classes: {class_names}")

    # Process each class folder
    for class_name in class_names:
        print(f"\nProcessing class: {class_name}")
        
        # Create destination directories for this class
        dest_train_class_dir = os.path.join(train_dir, class_name)
        dest_test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(dest_train_class_dir, exist_ok=True)
        os.makedirs(dest_test_class_dir, exist_ok=True)
        
        # Get a list of all image file paths for the current class
        class_path = os.path.join(INPUT_DIR, class_name)
        image_files = [
            os.path.join(class_path, f) for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ]
        
        if not image_files:
            print(f"  No images found in '{class_path}'. Skipping.")
            continue
            
        # Split the list of files into training and testing sets
        train_files, test_files = train_test_split(
            image_files,
            test_size=(1 - TRAIN_RATIO),
            random_state=RANDOM_STATE
        )
        
        print(f"  Splitting {len(image_files)} images into {len(train_files)} train and {len(test_files)} test.")
        
        # Copy the files to their new homes
        copy_files(train_files, dest_train_class_dir)
        copy_files(test_files, dest_test_class_dir)

    print("\nScript finished successfully!")
    print(f"Your data is now organized in the '{OUTPUT_DIR}' directory, ready for ImageFolder.")

if __name__ == "__main__":
    main()