import os

def fix_imports(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove the old get_tabpfn_classes definition
    import re
    pattern = re.compile(r'# (Dynamic importation helper|Dyn-imports helper) for TabPFN.*?return DummyTabPFNClassifier, DummyTabPFNEmbedding', re.DOTALL)
    content = pattern.sub('from dotpfn.utils.tabpfn_loader import get_tabpfn_classes', content)

    # In explain.py, it uses get_tabpfn_classes() - no change needed for calling it
    # In train.py, same thing. Wait, explain.py also has rom dotpfn.utils.tabpfn_loader import get_tabpfn_classes

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed {filepath}")

fix_imports('src/dotpfn/scripts/train.py')
fix_imports('src/dotpfn/scripts/explain.py')

# Also, we should add an option in get_tabpfn_classes() call in both files to use config.tabpfn.version if available
