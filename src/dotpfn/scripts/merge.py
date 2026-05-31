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
    # Define features to preserve explicitly based on user request
    PRESERVED_FEATURES = [
        'SEX', 'AGE', 'Height', 'Weight', 'BMI', 'PSQI', 'BDIII', 'ISI', 'ESS', 'SSS', 'STOPBANG', 
        'BQ', 'HTN', 'DM', 'DL', 'Liver', 'Kidney', 'Lung', 'IHD', 'CHF', 'Arrythmia', 'Cancer', 
        'Thyroid', 'Rhinitis', 'CTD', 'Epilepsy', 'CVA', 'Dementia', 'PSY', 'OSA_op', 'REM_Episodes', 
        'TST', 'Total_Stage_N1_ratio', 'Total_Stage_N2_ratio', 'Total_Stage_N3_ratio', 'Total_Stage_R_ratio', 
        'WASO', 'Sleep_latency', 'REM_sleep_latency', 'Sleep_Efficiency', 'Arousal_index', 'AHI', 
        'O2_satu_nadir', 'snoring_index', 'PLMs_index', 'PLMar_index', 'AHI_supine', 'AHI_REM', 
        'Apnea_NREM_avg', 'Apnea_REM_avg', 'Hypopnea_NREM_avg', 'Hypopnea_REM_avg', 'No_of_desatu', 
        'Min_O2_NREM', 'Min_O2_REM', 'HRV_NREM', 'HRV_REM', 'CCI'
    ]
    
    # Identify target labels (anything starting with adherence_)
    labels_to_keep = [col for col in metadata_df.columns if col.startswith('adherence_')]

    # Columns to preserve: ID, embedding_file, all labels, and requested features (if they exist)
    cols_to_keep = ['ID', 'embedding_file'] + labels_to_keep + [f for f in PRESERVED_FEATURES if f in metadata_df.columns]
    
    final_df = combined_df[cols_to_keep]

    # Save to a new CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Merge complete! Combined dataset saved to: {output_path}")
