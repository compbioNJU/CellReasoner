import argparse
import os
import scanpy as sc
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import issparse

def main():
    parser = argparse.ArgumentParser(description="Extract count and metadata from a h5ad file")
    parser.add_argument("--input_file", help="Input h5ad file path", required=True)
    parser.add_argument("--output_path", help="Output directory", required=True)
    parser.add_argument("--ranked_hvg_list", default='./data/ranked_hvg.list',help="ranked_hvg_list file", required=True)
    args = parser.parse_args()
    out_path = args.output_path
    gene_list_file = args.ranked_hvg_list
    os.makedirs(out_path, exist_ok=True)

    with open(gene_list_file, 'r') as f:
        gene_list = [line.strip() for line in f]

    prefix = os.path.basename(args.input_file).replace(".h5ad", "")

    print(f"\n{'='*100}")
    print(f"Processing {args.input_file}")
    print(f"{'='*100}")

    adata = sc.read_h5ad(args.input_file)

    available_genes = adata.var_names.tolist()
    matched_genes = [gene for gene in gene_list if gene in available_genes]

    print(f"Available genes: {len(available_genes)}")
    print(f"Matched {len(matched_genes)}/{len(gene_list)} genes")

    if matched_genes:

        print("Extracting matrix...")
        count_data = adata[:, matched_genes].X
        count_path = os.path.join(out_path, f"{prefix}.h5")
        save_h5(count_data, adata.obs_names.tolist(), matched_genes, count_path, data_type="count")


        print("Extracting metadata...")
        metadata = adata.obs
        meta_path = os.path.join(out_path, f"{prefix}.meta.csv")
        metadata.to_csv(meta_path)
        print(f"Saved metadata to {meta_path}")
    else:
        print("❌ No matched genes found. Skipping save.")

def save_h5(data, cell_names, gene_names, file_path, data_type):
    if issparse(data):
        data = data.toarray()

    with h5py.File(file_path, 'w') as h5f:
        grp = h5f.create_group(data_type)
        grp.create_dataset('data', data=data, chunks=True, compression='gzip')
        grp.create_dataset('cell_names', data=np.array(cell_names, dtype='S'))
        grp.create_dataset('gene_names', data=np.array(gene_names, dtype='S'))

    print(f"✔ Saved {data_type} matrix to {file_path}")

if __name__ == "__main__":
    """
    python s0.process_h5ad.py --input_file ./output/pbmc_demo.h5ad \
    --out_path ./output/ \
    --gene_list_file ./output/pbmc_demo.genes.txt
    """
    main()
