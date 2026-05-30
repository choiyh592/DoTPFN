import sys
import os
import argparse

# Guarantee local src/ package resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dotpfn.scripts.extract import run_extraction
from dotpfn.scripts.merge import run_merge

def main():
    parser = argparse.ArgumentParser(description="DoTPFN Document Indexing and Processing Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # Sub-command: extract
    extract_parser = subparsers.add_parser("extract", help="Extract document embeddings using ColQwen2")
    extract_parser.add_argument("--input_folder", type=str, required=True, help="Path to raw hypnogram image PNGs")
    extract_parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save tensors and index CSV")
    extract_parser.add_argument("--device", type=str, default="cpu", help="Compute device (e.g. 'cpu', 'cuda')")
    extract_parser.add_argument("--model_path", type=str, default="vidore/colqwen2-v0.1", help="Pretrained colpali/colqwen model checkpoint")

    # Sub-command: merge
    merge_parser = subparsers.add_parser("merge", help="Merge extracted document index with patient clinical metadata CSV")
    merge_parser.add_argument("--images_path", type=str, required=True, help="Path to ColQwen index CSV file (embedding_index.csv)")
    merge_parser.add_argument("--metadata_path", type=str, required=True, help="Path to raw patient clinical metadata CSV")
    merge_parser.add_argument("--output_path", type=str, required=True, help="Output file path for merged CSV")

    args = parser.parse_args()

    if args.command == "extract":
        run_extraction(
            input_folder=args.input_folder,
            output_dir=args.output_dir,
            device=args.device,
            model_path=args.model_path
        )
    elif args.command == "merge":
        run_merge(
            images_path=args.images_path,
            metadata_path=args.metadata_path,
            output_path=args.output_path
        )

if __name__ == "__main__":
    main()
