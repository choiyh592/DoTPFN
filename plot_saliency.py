import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from colpali_engine.models import ColQwen2Processor

import torch.nn as nn
from src.crossattentionpooler import CrossAttentionWithLearnedQueries

class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8):
        super().__init__()
        self.pooler = CrossAttentionWithLearnedQueries(
            input_dim=embed_dim, embed_dim=embed_dim, num_heads=num_heads
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        pooled_features = self.pooler(x)
        logits = self.classifier(pooled_features)
        return logits.squeeze(-1)

class DocumentCAMVisualizer:
    def __init__(self, model_weights_path: str, embed_dim: int = 128, device: str = "cpu"):
        self.device = torch.device(device)
        
        # Load the trained Attention Classifier
        self.classifier = AttentionClassifier(embed_dim=embed_dim).to(self.device).to(torch.float32)
        self.classifier.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.classifier.eval()
        
        # Load the ColQwen2 processor to figure out the dynamic grid sizes
        print("Loading processor to handle dynamic Qwen2 grid sizes...")
        self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

    def generate_1d_gradcam(self, embeddings: torch.Tensor) -> np.ndarray:
        """Computes importance weights for each patch using gradients."""
        # embeddings shape expected: [num_patches, 128]
        embeddings = embeddings.clone().detach().to(self.device, dtype=torch.float32).requires_grad_(True)
        
        # Forward pass (needs unsqueeze to simulate batch size of 1)
        logits = self.classifier(embeddings.unsqueeze(0))
        
        # Backward pass from the raw logit
        self.classifier.zero_grad()
        logits.backward()
        
        # 1. Get gradients and activations
        grads = embeddings.grad  # [num_patches, 128]
        acts = embeddings.detach() # [num_patches, 128]
        
        # 2. Pool gradients across the embedding dimension to get feature weights
        weights = grads.mean(dim=0) # [128]
        
        # 3. Compute CAM: weighted sum of activations
        cam = torch.matmul(acts, weights) # [num_patches]
        
        # 4. ReLU to keep only positive influences, then normalize
        cam = torch.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        return cam.cpu().numpy()

    def get_grid_shape(self, image: Image.Image) -> tuple:
        """Determines the dynamic spatial grid size of Qwen2 patches."""
        batch_doc = self.processor.process_images([image])
        
        # Qwen2-VL keeps grid info in 'image_grid_thw'. 
        # Format is typically (Time, Height, Width) per image.
        if hasattr(batch_doc, 'image_grid_thw'):
            grid_info = batch_doc.image_grid_thw[0] # Take first image in batch
            # grid_info is [1, grid_h, grid_w] for static images
            return int(grid_info[1]), int(grid_info[2])
        else:
            raise ValueError("Processor did not return spatial grid sizes.")

    def visualize(self, image_path: str, embedding_path: str, save_path: str = None):
        """Generates CAM, reshapes it using Qwen2 2x2 merging, and overlays it."""
        # 1. Load original Image & calculate expected ViT grid
        image = Image.open(image_path).convert("RGB")
        vit_grid_h, vit_grid_w = self.get_grid_shape(image)
        
        # 2. Calculate the LLM Grid (Qwen2 merges 2x2 patches)
        llm_grid_h = vit_grid_h // 2
        llm_grid_w = vit_grid_w // 2
        expected_image_tokens = llm_grid_h * llm_grid_w  # e.g., 31 * 24 = 744
        
        # 3. Load the embedding
        embeddings = torch.load(embedding_path, map_location=self.device)
        if embeddings.dim() == 3 and embeddings.size(0) == 1:
            embeddings = embeddings.squeeze(0)  # [num_patches, 128]
            
        num_tokens = embeddings.shape[0]
        
        # 4. Generate 1D CAM sequence
        cam_1d = self.generate_1d_gradcam(embeddings)
        
        # 5. Isolate the actual image tokens from the prompt/special tokens
        if num_tokens > expected_image_tokens:
            extra_tokens = num_tokens - expected_image_tokens
            
            # In Qwen2-VL, image tokens are usually sandwiched inside the prompt.
            # If your embeddings were saved with standard ColQwen2 processing,
            # the sequence is typically: [prompt/start tokens] + [image tokens] + [end tokens].
            # A common offset for the start of image tokens in Qwen2 is index 2 or 3. 
            # We will use a safe slicing method by finding the start index dynamically 
            # or relying on standard token boundaries. 
            
            # Assuming standard chat template: <|im_start|>user\n<|vision_start|> (4 tokens)
            # Adjust `start_idx` if your map looks misaligned or shifted!

            # ADDITION: From ColQwen2 Implementations, we can see that:
            # visual_prompt_prefix: ClassVar[str] = (
            #     "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
            # )
            # for which we can see that 4 tokens are correct.

            start_idx = 4
            
            # Safety check: ensure we don't slice out of bounds
            if start_idx + expected_image_tokens > num_tokens:
                start_idx = num_tokens - expected_image_tokens # Fallback to the end
                
            image_cam_1d = cam_1d[start_idx : start_idx + expected_image_tokens]
            
            print(f"Total tokens: {num_tokens}. Isolated {expected_image_tokens} image tokens starting at index {start_idx}.")
        elif num_tokens == expected_image_tokens:
            image_cam_1d = cam_1d
        else:
            raise ValueError(f"Not enough tokens! Expected at least {expected_image_tokens}, got {num_tokens}.")

        # 6. Reshape to the 2D Merged Spatial Map (e.g., 31x24)
        cam_2d = image_cam_1d.reshape(llm_grid_h, llm_grid_w)
        
        # 7. Overlay Process Using OpenCV
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h_orig, w_orig, _ = img_cv.shape
        
        # Resize the 31x24 heatmap beautifully back up to the full document size
        cam_resized = cv2.resize(cam_2d, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        
        # --- 1. HARD THRESHOLDING ---
        threshold = 0.25 # Adjust between 0.1 and 0.4 based on your preference
        
        # Any value below the threshold becomes exactly 0.0
        cam_resized[cam_resized < threshold] = 0.01 
        
        # --- 2. APPLY COLORMAP ---
        # Note: In the JET colormap, 0.0 becomes solid dark blue.
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # --- 3. SMART BLENDING (Transparent Background) ---
        # If you don't want the whole document covered in a dark blue wash,
        # we tell OpenCV to ONLY blend the pixels that survived the threshold.
        mask = (cam_resized > 0)[..., np.newaxis] 
        
        alpha = 0.5
        blended = cv2.addWeighted(img_cv, 1 - alpha, heatmap, alpha, 0)
        
        # Where mask is True, show the heatmap. Where False, show original document.
        overlay = np.where(mask, blended, img_cv)
        
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # 8. Plot and Save
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(image)
        axes[0].set_title("Original Document")
        axes[0].axis('off')
        
        axes[1].imshow(overlay_rgb)
        axes[1].set_title("Attention Activation Map (Grad-CAM)")
        axes[1].axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

# --- Execution ---
if __name__ == "__main__":
    # Update these paths based on your environment
    MODEL_WEIGHTS = "/home/yhchoi/PSG_DocParse_260501/experiments/best_model_label_adherence_5yr_fold_0.pt"
    SAMPLE_IMAGE = "/home/yhchoi/PSG_2025/260325_PSG_LastPages/PSG_Lastpage_Images_HypnogramMatch/10933307_2023_10.png"
    SAMPLE_EMBEDDING = "/home/yhchoi/PSG_DocParse_260501/embedding_files/10933307_2023_10.pt"
    
    visualizer = DocumentCAMVisualizer(model_weights_path=MODEL_WEIGHTS, device="cuda")
    visualizer.visualize(SAMPLE_IMAGE, SAMPLE_EMBEDDING, save_path="./cam_overlay.png")