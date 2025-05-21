import os
from pdb import run
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
import datetime
import builtins
import time
import psutil
from utils import run_inference


original_print = builtins.print
def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    original_print(f"{current_time} -", *args, **kwargs)

builtins.print = custom_print  


def main():
    start_time = time.time()
    print("-" * 80)
    print("Python script command:", " ".join(psutil.Process().cmdline()))
    print("-" * 80)
    
    print("Starting the inference...")
    
    argparser = argparse.ArgumentParser(description="Evaluate the results of Qell")
    argparser.add_argument("--model", type=str, required=True, default="CellReasoner-7B", choices=["CellReasoner-7B","CellReasoner-32B"], help="Path to the model")
    argparser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation")
    argparser.add_argument('--reasoning_mode', action='store_true', help='Use expert reasoning mode (with <think></think> in instruction)')
    argparser.add_argument("--input_json", type=str, required=True, help="Path to the Input JSON file")
    argparser.add_argument("--output_path", type=str, required=True, help="Path to store the results")
    argparser.add_argument("--has_label", action='store_true', help="Whether the input JSON file contains cell type information")
    
    args = argparser.parse_args()
    reasoning_mode = args.reasoning_mode
    eval_file_path = args.input_json
    data_name = os.path.basename(eval_file_path).split('.')[0].replace('_for_CellReasoner', '')
    output_path = args.output_path
    batch_size = args.batch_size
    model_name = args.model
    model_path = f"guangshuo/{model_name}"
    has_label = args.has_label
    
    os.makedirs(output_path, exist_ok=True)

    print(f"Model path: {model_path}")
    
    
    data =  pd.read_json(eval_file_path)
    print(f"Loaded data from {eval_file_path}, Number of samples: {len(data)}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",    
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Model and tokenizer loaded successfully.")
    output_data = run_inference(
        model=model,
        tokenizer=tokenizer,
        data=data,
        batch_size=batch_size,
        log_interval=10,
        reasoning_mode=reasoning_mode,
        has_label=has_label,
    )
    print("Inference finished.")
    
    torch.cuda.empty_cache()
    output_data_df = pd.DataFrame(output_data)
    result_path = os.path.join(output_path, f"{data_name}_CellReasoner_result.csv")
    output_data_df.to_csv(result_path, index=False)
    print(f"Saved results to {result_path}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Script execution time: {execution_time:.4f} seconds. \n")
    
    
if __name__ == "__main__":
    """
    pyton predict_per_cell_with_embedding.py \
    --model_name "Qell-7B" \
    --output_path "/home/shy/deepseek_cancer/cgs_script/output" \
    --eval_stratecy "align" \
    --eval_json "/home/shy/deepseek_cancer/test_data/multi_omics/pbmc/pbmc_atac/pbmc_atac_all_cell.json" \
    --generate_cell_embedding 
    """
    
    main()
    
    
    
    