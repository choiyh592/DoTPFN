import torch
import csv
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColQwen2, ColQwen2Processor

class ColQwenIndexer:
    def __init__(self, model_path: str = "vidore/colqwen2-v0.1", device: str = "cpu"):
        self.device = device
        self.processor = ColQwen2Processor.from_pretrained(model_path)
        self.model = ColQwen2.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device,
        ).eval()

    @torch.no_grad()
    def process_folder(self, input_folder: str, output_dir: str):
        input_path = Path(input_folder)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create a subfolder for the actual tensor files
        embeddings_dir = output_path / "embedding_files"
        embeddings_dir.mkdir(exist_ok=True)

        # 1. Find all PNGs
        image_files = list(input_path.glob("*.png"))
        if not image_files:
            print(f"No PNG files found in {input_folder}")
            return

        index_data = []

        # 2. Process images
        print(f"Processing {len(image_files)} images...")
        for img_file in tqdm(image_files):
            try:
                # Load image
                image = Image.open(img_file).convert("RGB")
                
                # Extract Embeddings
                batch_doc = self.processor.process_images([image]).to(self.model.device)
                embeddings = self.model(**batch_doc) # Shape: [1, num_patches, 128]
                
                # Save individual embedding file
                emb_filename = f"{img_file.stem}.pt"
                emb_path = embeddings_dir / emb_filename
                torch.save(embeddings.cpu(), emb_path)

                # Record for CSV
                index_data.append({
                    "image_name": img_file.name,
                    "image_path": str(img_file.absolute()),
                    "embedding_file": str(emb_path.absolute())
                })
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")

        # 3. Save CSV Index
        csv_path = output_path / "embedding_index.csv"
        self._write_csv(csv_path, index_data)
        print(f"\nProcessing complete!")
        print(f"Index saved to: {csv_path}")

    def _write_csv(self, path: Path, data: list):
        keys = ["image_name", "image_path", "embedding_file"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

# --- Execution ---
if __name__ == "__main__":
    # Configure your paths here
    SUBFOLDER_PATH = "/home/won_ju_kim/yhchoi/PSG_260408/PSG_Lastpage_Images_HypnogramMatch" 
    OUTPUT_FOLDER = "./output_embeddings"
    
    # Initialize and Run
    indexer = ColQwenIndexer(device="cuda") # Change to "cuda" for GPU speed
    indexer.process_folder(SUBFOLDER_PATH, OUTPUT_FOLDER)