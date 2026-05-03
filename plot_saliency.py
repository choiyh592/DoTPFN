import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from colpali_engine.models import ColQwen2Processor

# Import your custom classifier from your existing script
from classifier import AttentionClassifier 

class DocumentCAMVisualizer:
    def __init__(self, model_weights_path: str, embed_dim: int = 128, device: str = "cpu"):
        self.device = torch.device(device)
        
        # Load the trained Attention Classifier
        self.classifier = AttentionClassifier(embed_dim=embed_dim).to(self.device)
        self.classifier.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.classifier.eval()
        
        # Load the ColQwen2 processor to figure out the dynamic grid sizes
        print("Loading processor to handle dynamic Qwen2 grid sizes...")
        self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

    def generate_1d_gradcam(self, embeddings: torch.Tensor) -> np.ndarray:
        """Computes importance weights for each patch using gradients."""
        # embeddings shape expected: [num_patches, 128]
        embeddings = embeddings.clone().detach().to(self.device).requires_grad_(True)
        
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
        """Generates CAM, reshapes it, and overlays it on the image."""
        # 1. Load original Image & calculate expected 2D grid
        image = Image.open(image_path).convert("RGB")
        grid_h, grid_w = self.get_grid_shape(image)
        
        # 2. Load the specific embedding tensor for this image
        embeddings = torch.load(embedding_path, map_location=self.device)
        if embeddings.dim() == 3 and embeddings.size(0) == 1:
            embeddings = embeddings.squeeze(0) # [num_patches, 128]
            
        # Verify patch count matches grid
        num_patches = embeddings.shape[0]
        if grid_h * grid_w != num_patches:
            print(f"Warning: Grid size {grid_h}x{grid_w} ({grid_h*grid_w}) doesn't match patch count {num_patches}")
            
        # 3. Generate 1D CAM sequence
        cam_1d = self.generate_1d_gradcam(embeddings)
        
        # 4. Reshape to 2D Spatial Map
        # Note: If patching is flattened row-major, simple reshape works.
        cam_2d = cam_1d.reshape(grid_h, grid_w)
        
        # 5. Overlay Process Using OpenCV
        # Convert PIL to CV2 format (RGB to BGR)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h_orig, w_orig, _ = img_cv.shape
        
        # Resize the tiny heatmap to the original document size
        cam_resized = cv2.resize(cam_2d, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap (JET is standard for heatmaps)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # Blend the heatmap and original image
        alpha = 0.5 # Opacity of the heatmap
        overlay = cv2.addWeighted(img_cv, 1 - alpha, heatmap, alpha, 0)
        
        # Convert back to RGB for matplotlib/saving
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # 6. Plot and Save
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
    MODEL_WEIGHTS = "/home/won_ju_kim/yhchoi/PSG_260408/model_weights_1.pth"
    SAMPLE_IMAGE = "./documents_png/sample_document.png"
    SAMPLE_EMBEDDING = "./output_embeddings/embedding_files/sample_document.pt"
    
    visualizer = DocumentCAMVisualizer(model_weights_path=MODEL_WEIGHTS, device="cuda")
    visualizer.visualize(SAMPLE_IMAGE, SAMPLE_EMBEDDING, save_path="./cam_overlay.png")