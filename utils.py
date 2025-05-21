import logging
import os
import numpy as np
import torch
import random
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import torch
from tqdm import tqdm

def run_inference(
    model,
    tokenizer,
    data,
    batch_size=4,
    log_interval=1,
    reasoning_mode=False,
    has_label=True
):
    """
    Batch inference wrapper that returns a list of output data.

    Args:
        model: The loaded HuggingFace model for inference.
        tokenizer: Corresponding tokenizer for the model.
        data: A DataFrame containing 'input', 'instruction', and 'cell_name' (optional 'output').
        batch_size: Number of samples per batch.
        temperature: Sampling temperature for generation.
        top_p: Nucleus sampling threshold (top-p).
        log_interval: Print progress every N batches.
        reasoning_mode: Whether to prepend the <think> tag to enable reasoning mode.
        has_label: Whether the input data includes ground truth labels in the 'output' column.

    Returns:
        output_data: A list of dictionaries, each containing the model's output and related metadata.
    """

    model.eval()
    tokenizer.padding_side = 'left'
    
    if reasoning_mode:
        temperature = 0.6
        top_p = 0.4
    else:
        temperature = 0.01
        top_p = 0.01
    

    output_data = []
    responses = []

    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    for index_batch, batch in enumerate(batches):
        if index_batch % log_interval == 0:
            print(f"Processing batch {index_batch+1}/{len(batches)}")

        texts, labels, inputs, instructions, cell_names = [], [], [], [], []

        for _, cell in batch.iterrows():
            _input = cell['input']
            _instruction = cell['instruction']
            messages = [
                {"role": "system", "content": _instruction},
                {"role": "user", "content": _input}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            text = text.replace('<think>', '')
            if reasoning_mode:
                text += '<think> '

            texts.append(text)
            inputs.append(_input)
            instructions.append(_instruction)
            cell_names.append(cell['cell_name'])
            if has_label:
                labels.append(cell['output'])

        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
            return_attention_mask=True
        ).to(model.device)

        generation_config = {
            "max_new_tokens": 4096,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            "pad_token_id": tokenizer.eos_token_id,
            "attention_mask": model_inputs.attention_mask,
        }

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=model_inputs.input_ids,
                **generation_config
            )

        for i in range(len(model_inputs.input_ids)):
            input_length = model_inputs.input_ids[i].shape[0]
            output_ids = generated_ids.sequences[i][input_length:]
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            responses.append(response)

            record = {
                "cell_name": cell_names[i],
                "response": response
            }
            if has_label:
                record["labels"] = labels[i]
            output_data.append(record)

    print("Inference finished.")
    return output_data



def set_seed(seed=1234):
    '''
    seed: int, random seed
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def plot_score_violin(
    all_cell_df: pd.DataFrame,
    show_mean_point: bool = True,
    show_mean_text: bool = True,
    show_text: bool = True,
    fig_size: tuple = (10, 6),
    sort: str = 'ascend',
    custom_palette=None,
    save_path: str = None,
):

    score_cols = [c for c in all_cell_df.columns if c.endswith('Score')]
    long_df = all_cell_df[score_cols].melt(var_name='Model', value_name='Score')
    mean_scores = long_df.groupby('Model')['Score'].mean().reset_index()

    if sort == 'ascend':
        model_order = mean_scores.sort_values('Score')['Model'].tolist()
    elif sort == 'descend':
        model_order = mean_scores.sort_values('Score', ascending=False)['Model'].tolist()
    else:
        model_order = None

    if custom_palette is None:
        custom_palette = ['#5e9cef', '#5fb668', '#919423', '#d55b54', '#d74cea', '#ff8eb6']
    num_scores = len(score_cols)

    plt.figure(figsize=fig_size)
    ax = sns.violinplot(
        x='Model', y='Score', data=long_df,
        inner=None, order=model_order, palette=custom_palette
    )

    if model_order:
        mean_scores = mean_scores.set_index('Model').loc[model_order].reset_index()

    if show_mean_point:
        for i, row in mean_scores.iterrows():
            color = custom_palette[i] if i < len(custom_palette) else 'black'
            plt.scatter(i, row['Score'], color='black', marker='o', zorder=3)

    if show_text and show_mean_text:
        for i, row in mean_scores.iterrows():
            plt.text(
                i, row['Score'] + 0.02,
                f"{row['Score']:.2f}",
                ha='center', va='bottom', fontsize=10, color='black'
            )

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    if show_text:
        plt.xticks(rotation=90)
        new_labels = [re.sub(r'_score$', '', label.get_text()) for label in ax.get_xticklabels()]
        ax.set_xticklabels(new_labels)  
        plt.xlabel('Model')

    if not show_text:
        plt.xticks([])  

    plt.yticks([0, 0.25, 0.5, 1.0])

    # plt.title('Violin Plot of Scores with Exact Mean Points')
    plt.ylabel('Score')
    # plt.xlabel('Model')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved figure to {save_path}")

    plt.show()



def clean_series(s: pd.Series) -> pd.Series:
    return s.astype(str) \
            .str.replace(r'\s+', ' ', regex=True) \
            .str.strip()


def plot_cross_label_heatmap(
    df,
    true_label_col,
    pred_label_col,
    normalize=True,
    threshold=None,
    save_path=None
):
    df[true_label_col] = clean_series(df[true_label_col])
    df[pred_label_col] = clean_series(df[pred_label_col])
    if normalize and threshold is not None:
        cross_tab = pd.crosstab(df[true_label_col], df[pred_label_col], normalize='index')

        valid_pairs = cross_tab.stack()[cross_tab.stack() >= threshold].reset_index()
        valid_true_labels = valid_pairs[true_label_col].unique()
        valid_pred_labels = valid_pairs[pred_label_col].unique()

        df = df[df[true_label_col].isin(valid_true_labels) & df[pred_label_col].isin(valid_pred_labels)]

    if normalize:
        cross_tab = pd.crosstab(df[true_label_col], df[pred_label_col], normalize='index')
        fmt = ".2f"
    else:
        cross_tab = pd.crosstab(df[true_label_col], df[pred_label_col])
        fmt = "d"

    plt.figure(figsize=(len(cross_tab.columns)*1, len(cross_tab.index)*1))
    sns.heatmap(
        cross_tab,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
    )
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.title(
        "Cross-Label Mapping: True vs. Predicted (Row Normalized)"
        if normalize else
        "Cross-Label Counts: True vs. Predicted",
        fontsize=14
    )
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_score_stacked_bar(
    all_cell_df: pd.DataFrame,
    fig_size=(12, 6),
    ylabel='Proportion',
    title='Stacked Bar Plot of Score Distributions',
    score_suffix='Score',
    score_mapping={1: 'Subtype correct', 0.5: 'Major correct', 0.25: 'Partially correct', 0: 'Incorrect'},
    score_order=None,
    model_order=None,
    score_colors=None,
    show_mean_text=True, 
    show_mean_line=True, 
    save_path=None
):
    if score_colors is None:
        score_colors = {
            'Subtype correct': '#1F77B4',
            'Major correct':   '#59B0DE',
            'Partially correct': '#95E1D3',
            'Incorrect':       '#FFA95D'
        }

    score_cols = [c for c in all_cell_df.columns if c.endswith(score_suffix)]

    if score_order is None:
        score_order = list(score_mapping.values())

    dist_dict = {}
    means_dict = {}
    for col in score_cols:

        vc = all_cell_df[col].value_counts(normalize=True)

        mean_value = all_cell_df[col].mean()
        means_dict[col] = mean_value
        vc.index = vc.index.astype(float)
        vc = vc.rename(index=score_mapping)
        vc = vc.reindex(score_order, fill_value=0)

        dist_dict[col] = vc

    score_distribution = pd.DataFrame(dist_dict)

    score_distribution = score_distribution.T

    if model_order is not None:
        score_distribution = score_distribution.reindex(model_order)

    plt.figure(figsize=fig_size)
    bottom = None
    for label in score_order:
        bars = plt.bar(
            score_distribution.index,
            score_distribution[label],
            bottom=bottom,
            label=label,
            color=score_colors.get(label)
        )
        bottom = score_distribution[label] if bottom is None else bottom + score_distribution[label]

    if show_mean_text:
        for i, model in enumerate(score_distribution.index):
            total_height = bottom[i]
            mean_y_position = means_dict[model] 
            plt.text(i, mean_y_position + 0.02, f'{means_dict[model]:.2f}', ha='center', va='bottom',
                     fontsize=10, color='black')

    if show_mean_line:
        for i, model in enumerate(score_distribution.index):
            total_height = bottom[i]
            mean_y_position = means_dict[model] 
            plt.plot([i - 0.4, i + 0.4], [mean_y_position, mean_y_position], color='#c00000', linestyle='--', lw=1)
    
    sns.despine(left=False, bottom=False)

    if show_mean_text: 
        labels = [label.get_text() for label in plt.gca().get_xticklabels()] 
        new_labels = [re.sub(r'_score$', '', label) for label in labels]  
        plt.gca().set_xticklabels(new_labels)  

    plt.ylabel(ylabel)
    # plt.title(title)
    plt.xticks(rotation=90) 
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved figure to {save_path}")

    plt.show()
