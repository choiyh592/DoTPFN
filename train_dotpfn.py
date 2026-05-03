from tabpfn import TabPFNClassifier

# DEMO

# 1. Load the official Pretrained weights into memory
official_tabpfn = TabPFNClassifier()
# (You might need to call a dummy .fit() here if PriorLabs lazy-loads the weights)

# Extract the base PyTorch architecture (adjust attribute name if needed)
base_pytorch_model = official_tabpfn.model_

# 2. Instantiate MMPFN
# Assuming your text/image encoder outputs 768-dim embeddings (like DINOv2 or ELECTRA)
mmpfn = MultiModalPFN(
    pretrained_tabpfn=base_pytorch_model,
    non_tab_dim=768, 
    n_heads=32, 
    k_tokens=24
)

# Move to GPU and set to training mode
mmpfn = mmpfn.to('cuda')
mmpfn.train()

# --- Dummy Training Step ---
# Ri (Total Rows) = 100. Batch Size = 2. Features = 10.
x_tab = torch.randn(100, 2, 10).to('cuda')

# Labels for the first 80 rows (Support Set). Test set is the remaining 20 rows.
y = torch.randint(0, 2, (80,)).to('cuda') 

# Pre-extracted document CLS embeddings for all 100 rows
x_non_tab = torch.randn(2, 100, 768).to('cuda')

# Forward pass
logits = mmpfn(x_tab, y, x_non_tab)

# Calculate loss against your true test labels (the last 20 rows)
# test_labels = ...
# loss = cross_entropy_loss(logits, test_labels)
# loss.backward()