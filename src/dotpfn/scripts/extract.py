import torch
import csv
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColQwen2, ColQwen2Processor
from dotpfn.utils.logging import setup_logger

logger = setup_logger("DoTPFN.Extract")

class ColQwenIndexer:
    def __init__(self, model_path: str = "vidore/colqwen2-v0.1", device: str = "cpu"):
        self.device = device
        logger.info(f"Loading ColQwen2 processor from {model_path}...")
        self.processor = ColQwen2Processor.from_pretrained(model_path)
        logger.info(f"Loading ColQwen2 model on {device}...")
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
        
        embeddings_dir = output_path / "embedding_files"
        embeddings_dir.mkdir(exist_ok=True)

        image_files = list(input_path.glob("*.png"))
        if not image_files:
            logger.warning(f"No PNG files found in {input_folder}")
            return

        index_data = []
        logger.info(f"Processing {len(image_files)} images...")
        for img_file in tqdm(image_files):
            try:
                image = Image.open(img_file).convert("RGB")
                batch_doc = self.processor.process_images([image]).to(self.model.device)
                embeddings = self.model(**batch_doc) # Shape: [1, num_patches, 128]
                
                emb_filename = f"{img_file.stem}.pt"
                emb_path = embeddings_dir / emb_filename
                torch.save(embeddings.cpu(), emb_path)

                index_data.append({
                    "image_name": img_file.name,
                    "image_path": str(img_file.absolute()),
                    "embedding_file": str(emb_path.absolute())
                })
            except Exception as e:
                logger.error(f"Error processing {img_file.name}: {e}")

        csv_path = output_path / "embedding_index.csv"
        self._write_csv(csv_path, index_data)
        logger.info(f"Processing complete! Index saved to: {csv_path}")

    def _write_csv(self, path: Path, data: list):
        keys = ["image_name", "image_path", "embedding_file"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
            
def run_extraction(input_folder: str, output_dir: str, device: str = "cpu", model_path: str = "vidore/colqwen2-v0.1"):
    indexer = ColQwenIndexer(model_path=model_path, device=device)
    indexer.process_folder(input_folder, output_dir)
