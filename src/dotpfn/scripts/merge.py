import pandas as pd
import os
from dotpfn.utils.logging import setup_logger

logger = setup_logger("DoTPFN.Merge")

def run_merge(images_path: str, metadata_path: str, output_path: str):
    logger.info(f"Loading image embedding index from {images_path}...")
    images_df = pd.read_csv(images_path)
    
    logger.info(f"Loading patient metadata from {metadata_path}...")
    metadata_df = pd.read_csv(metadata_path)

    # Extract ID from the image name (e.g. 10001_Page1.png -> 10001)
    logger.info("Extracting IDs from image names and merging datasets...")
    images_df['ID'] = images_df['image_name'].str.split('_').str[0].astype(int)

    # Left join to preserve every ID from the images CSV
    combined_df = pd.merge(images_df, metadata_df, on='ID', how='left')

    # Reorder columns: ID and embedding_file first, followed by other metadata
    cols_to_keep = ['ID', 'embedding_file'] + [col for col in metadata_df.columns if col != 'ID']
    final_df = combined_df[cols_to_keep]

    # Save to a new CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Merge complete! Combined dataset saved to: {output_path}")
