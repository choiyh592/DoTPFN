import os

merge_path = 'src/dotpfn/scripts/merge.py'
with open(merge_path, 'r', encoding='utf-8') as f:
    merge_code = f.read()

new_merge_logic = '''
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
'''

import re
merge_code = re.sub(
    r"cols_to_keep = \['ID', 'embedding_file'\] \+ \[col for col in metadata_df\.columns if col != 'ID'\]\s+final_df = combined_df\[cols_to_keep\]",
    new_merge_logic.strip(),
    merge_code
)

with open(merge_path, 'w', encoding='utf-8') as f:
    f.write(merge_code)
print("Updated merge.py")

# Now update infer.py to output metrics
infer_path = 'src/dotpfn/scripts/infer.py'
with open(infer_path, 'r', encoding='utf-8') as f:
    infer_code = f.read()

# Make sure compute_metrics is imported
if 'from dotpfn.utils.metrics import compute_metrics' not in infer_code:
    infer_code = infer_code.replace(
        'from dotpfn.utils.tabpfn_loader import get_tabpfn_classes',
        'from dotpfn.utils.tabpfn_loader import get_tabpfn_classes\nfrom dotpfn.utils.metrics import compute_metrics'
    )

# Add metrics logic at the end
metrics_logic = '''
    query_df["probability"] = probs
    query_df["prediction"] = (query_df["probability"] >= getattr(config.inference, "threshold", 0.5)).astype(int)
    
    # Calculate metrics if ground truth is available
    if config.data.target_label in query_df.columns:
        y_true = query_df[config.data.target_label].values
        # Filter out NaNs if any
        valid_idx = ~np.isnan(y_true)
        if valid_idx.sum() > 0:
            metrics = compute_metrics(y_true[valid_idx], probs[valid_idx])
            logger.info("=========================================")
            logger.info("INFERENCE METRICS")
            logger.info("=========================================")
            for k, v in metrics.items():
                logger.info(f"{k}: {v:.4f}")
            logger.info("=========================================")
        else:
            logger.warning("Target label column exists but contains only NaNs.")
    else:
        logger.info(f"Target label '{config.data.target_label}' not found in query CSV. Skipping metrics calculation.")

    out_path = config.inference.output_path
'''

infer_code = re.sub(
    r'query_df\["probability"\] = probs.*?out_path = config\.inference\.output_path',
    metrics_logic.strip(),
    infer_code,
    flags=re.DOTALL
)

with open(infer_path, 'w', encoding='utf-8') as f:
    f.write(infer_code)
print("Updated infer.py")

