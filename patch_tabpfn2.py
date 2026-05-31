import os

loader_path = 'src/dotpfn/utils/tabpfn_loader.py'
with open(loader_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Change the embedding_import for 'tabpfn' to None to prevent ImportError if it doesn't exist
content = content.replace(
    '"embedding_import": ("tabpfn.embedding", "TabPFNEmbedding"),',
    '"embedding_import": None,  # V2.6+ might not have an embedding API exposed'
)

# Also, let's make the error message more verbose so if it fails again we know why
import re
content = re.sub(
    r'logger\.debug\(f"Could not load backend.*?return None',
    'logger.error(f"Could not load backend \'{backend_name}\': {e}")\n        return None',
    content,
    flags=re.DOTALL
)

with open(loader_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Patched tabpfn_loader.py")
