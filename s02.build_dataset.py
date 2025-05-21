import datetime
import h5py
import pandas as pd
import numpy as np
import argparse
import os
import time
import json
import builtins
import time
import psutil

from utils import set_seed

original_print = builtins.print
def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    original_print(f"{current_time} -", *args, **kwargs)

builtins.print = custom_print  

set_seed(1234)
pd.set_option('future.no_silent_downcasting', True)

def process_gene_expression(gene_counts_file_path, seq_length=1000, pad_token='<PAD>'):
    start_time = time.time() 
    
    gene_names_list = []
    idx_list = []

    with h5py.File(gene_counts_file_path, 'r') as f:
        data = f['count/data'][:] 
        cell_names = f['count/cell_names'][:]  
        gene_names = f['count/gene_names'][:] 

        cell_names = [cell.decode('utf-8') for cell in cell_names]
        gene_names = [gene.decode('utf-8') for gene in gene_names]

    df = pd.DataFrame(data, index=cell_names, columns=gene_names)
    
    print(f"Start processing data... {df.shape}")
    print(f"The number of cells: {df.shape[0]}")

    for idx, row in df.iterrows():
        non_zero_genes = row.replace(0, np.nan).dropna().index.tolist()
        if len(non_zero_genes) > seq_length:
            top_genes = non_zero_genes[:seq_length]
            gene_names_list.append(top_genes)
            idx_list.append(idx)
        else:
            gene_names_list.append(non_zero_genes + [pad_token] * (seq_length - len(non_zero_genes)))
            idx_list.append(idx)
    
    print("Processing done!")
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    print(f"Total processing time: {total_time} seconds \n")
    
    return pd.DataFrame(gene_names_list, index=idx_list)

def main():
    """

    python s02.processed_data_for_deepseek.py --gene_counts_file_path ./output/pbmc_demo.h5 \
    --output_path ./output/ \
    --meta_file_path ./output/pbmc_demo.meta.csv \
    --cell_type_column "seurat_annotations" \
    --seq_length 1000 \
    """
    parser = argparse.ArgumentParser(description='Process gene expression data and build feature vectors')
    parser.add_argument('--h5_path', type=str, help='Path to the HDF5 file containing gene data')
    parser.add_argument('--output_path', type=str, help="Path to the output")
    parser.add_argument('--seq_length', type=int, default=1000, help='Length of the sequence')
    parser.add_argument('--meta_file_path', type=str, help='Path to the meta file (.meta.csv)')
    parser.add_argument('--cell_type_column', type=str, help='Cell type column name in the meta file (optional)')
    parser.add_argument('--reasoning_mode', action='store_true', help='Use expert reasoning mode (with <think></think> in instruction)')

    
    args = parser.parse_args()
    
    gene_file_path = args.h5_path
    output_dir = args.output_path
    # log_name = args.log_name
    seq_length = args.seq_length
    meta_file_path = args.meta_file_path
    cell_type_column = args.cell_type_column
    reasoning_mode = args.reasoning_mode
    
    os.makedirs(output_dir, exist_ok=True)
    print("Start processing gene expression data...")

    # Step 1: Process gene expression data
    gname_df = process_gene_expression(gene_file_path, seq_length=seq_length)
    base_name = os.path.basename(os.path.normpath(gene_file_path))
    base_name = base_name.split('.')[0]
    
    # Step 2: Read metadata and generate JSON file
    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(f"Meta file not found at {meta_file_path}")
    
    meta_df = pd.read_csv(meta_file_path, index_col=0)
    
    # Check if 'cell_type_column' exists in the meta file
    if cell_type_column and cell_type_column in meta_df.columns:
        cell_type_df = meta_df[[cell_type_column]].rename(columns={cell_type_column: 'cell_type'})
        gname_df['merge'] = gname_df.apply(lambda row: [x for x in row if x != '<PAD>'], axis=1)
        gname_df = gname_df[['merge']]
        gname_df = pd.merge(gname_df, cell_type_df, left_index=True, right_index=True, how='inner')
        gname_df['merge'] = gname_df['merge'].apply(lambda x: ', '.join(map(str, x)))
        gname_df = gname_df.rename(columns={'merge': 'cell_sentence'})
        include_output = True
    else:
        # If 'cell_type_column' is not provided or not found in the meta file, don't include the 'output' field
        gname_df['merge'] = gname_df.apply(lambda row: [x for x in row if x != '<PAD>'], axis=1)
        gname_df = gname_df[['merge']]
        gname_df['merge'] = gname_df['merge'].apply(lambda x: ', '.join(map(str, x)))
        gname_df = gname_df.rename(columns={'merge': 'cell_sentence'})
        include_output = False
    
    # Build JSON file
    if reasoning_mode:
        Insturction = "These are highly expressed genes within a certain type of tumor cells; please annotate the cell type based on these marker genes. Please infer the cell type based on the marker genes, and place the reason within the <think></think> tag."
    else:
        Insturction = "These are highly expressed genes within a certain type of tumor cells; please annotate the cell type based on these marker genes. Please infer the cell type based on the marker genes, and directly give the final annotation result."
    
    
    json_data = []
    for index, row in gname_df.iterrows():
        data = {
            "cell_name": index,
            "instruction": Insturction,
            "input": row['cell_sentence'],
        }
        if include_output:
            data["output"] = row['cell_type']
        json_data.append(data)
    
    output_file = os.path.join(output_dir, f'{base_name}_for_CellReasoner.json')
    with open(output_file, 'w') as f_out:
        json.dump(json_data, f_out, indent=4)

if __name__ == '__main__':
    main()
    print("Data processing completed.")
