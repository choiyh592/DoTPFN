import pandas as pd

IMAGES_PATH = '/home/won_ju_kim/yhchoi/PSG_260408/output_embeddings/embedding_index.csv' # Path to emb
METADATA_PATH = '/home/won_ju_kim/yhchoi/PSG_260408/Data_PSG_260324.csv' # Path to patient metadata
OUT_PATH = '/home/won_ju_kim/yhchoi/PSG_260408' # Path to output merged csv

# 1. Load your CSV files
images_df = pd.read_csv(IMAGES_PATH)
metadata_df = pd.read_csv(METADATA_PATH)

# 2. Extract the "ID" from the 'image_name' column
# We split by '_' and take the first element [0]
images_df['ID'] = images_df['image_name'].str.split('_').str[0].astype(int)

# 3. Merge the dataframes
# We use a 'left' join to ensure every ID from the images CSV is preserved
# even if there isn't a matching entry in the metadata CSV.
combined_df = pd.merge(images_df, metadata_df, on='ID', how='left')

# 4. Select and reorder the specific columns you requested
# We keep 'ID', 'embedding_file', and then all other metadata columns
# Note: This assumes metadata_df has columns other than 'ID'
cols_to_keep = ['ID', 'embedding_file'] + [col for col in metadata_df.columns if col != 'ID']
final_df = combined_df[cols_to_keep]

# 5. Save to a new CSV
final_df.to_csv(OUT_PATH + '/final_combined_data.csv', index=False)

print("Merge complete! Saved to final_combined_data.csv")