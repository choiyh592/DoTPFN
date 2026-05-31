import os
import re

infer_path = 'src/dotpfn/scripts/infer.py'
with open(infer_path, 'r', encoding='utf-8') as f:
    infer_code = f.read()

# Replace the query_df loading logic
new_query_logic = '''
    # Load query set (the new patients to predict on)
    logger.info(f"Loading query CSV from {config.data.query_csv_path}")
    query_df = pd.read_csv(config.data.query_csv_path)
    
    # Filter query_df for labels as in the training script
    if config.data.target_label in query_df.columns:
        query_df = query_df.dropna(subset=[config.data.target_label]).copy()
        query_df = query_df[query_df[config.data.target_label].isin([0, 1])].reset_index(drop=True)
        
    condition_on = getattr(config.data, 'condition_on', None)
    if condition_on is not None and condition_on in query_df.columns:
        query_df = query_df[query_df[condition_on] == 1].reset_index(drop=True)

    if len(query_df) == 0:
        logger.error("Query CSV is empty after filtering!")
        return
        
    query_doc_embs = load_all_document_embeddings(query_df)

    if model_type in ["multimodal", "image_pfn"]:
        # Need support set for in-context learning
        logger.info(f"Loading support CSV from {config.data.support_csv_path}")
        support_df = pd.read_csv(config.data.support_csv_path)
        support_df = support_df.dropna(subset=[config.data.target_label])
        support_df = support_df[support_df[config.data.target_label].isin([0, 1])].reset_index(drop=True)
        
        if condition_on is not None and condition_on in support_df.columns:
            support_df = support_df[support_df[condition_on] == 1].reset_index(drop=True)
'''

# Find the block from "logger.info(f"Loading query CSV" to "support_df = support_df[support_df[config.data.target_label].isin([0, 1])].reset_index(drop=True)"
pattern = re.compile(
    r'# Load query set.*?support_df = support_df\[support_df\[config\.data\.target_label\]\.isin\(\[0, 1\]\)\]\.reset_index\(drop=True\)', 
    re.DOTALL
)

infer_code = pattern.sub(new_query_logic.strip(), infer_code)

with open(infer_path, 'w', encoding='utf-8') as f:
    f.write(infer_code)
print("Updated infer.py with label filtering")
