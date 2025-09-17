import os
import pandas as pd

def create_hsi_csv(data_dir='data/processed', output_csv='hsi_dataset.csv'):
    """
    Creates a CSV file containing image paths and their corresponding classes
    from the HSI dataset structure.

    The expected directory structure is:
    data_dir/
    ├── class_name_1/
    │   ├── scan_name_1/
    │   │   └── CWT_morl/
    │   │       ├── image1.png
    │   │       └── image2.png
    │   └── scan_name_2/
    │       └── CWT_morl/
    │           └── ...
    └── class_name_2/
        └── ...

    Args:
        data_dir (str): The root directory of the processed data.
        output_csv (str): The path for the output CSV file.
    """
    image_data = []
    
    # Check if the data directory exists
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    # Get class names from the subdirectories in data_dir and sort them
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        
        # Get scan names from the subdirectories in each class folder
        scan_names = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
        
        for scan_name in scan_names:
            cwt_morl_path = os.path.join(class_path, scan_name, 'CWT_morl')
            
            if os.path.isdir(cwt_morl_path):
                # List all files in the CWT_morl directory
                for image_name in os.listdir(cwt_morl_path):
                    image_path = os.path.join(cwt_morl_path, image_name)
                    # Add the relative path and class to our list
                    image_data.append([image_path, class_name])

    if not image_data:
        print("No images found. The CSV file will not be created.")
        return

    # Create a pandas DataFrame and save it to a CSV file
    df = pd.DataFrame(image_data, columns=['image_path', 'class'])
    df.to_csv(output_csv, index=False)
    print(f"Successfully created '{output_csv}' with {len(df)} entries.")

if __name__ == '__main__':
    create_hsi_csv()
