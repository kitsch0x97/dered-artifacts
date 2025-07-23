import torch
import torch.nn as nn
from model.GCNConv import FCGPairDataset, FCGPairDataset_test, SiameseGCNConv
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.DataLoader import create_dataloaders, BalancedWrapper
import torch.nn.functional as F
from sklearn.utils import resample
from tqdm import tqdm  
import numpy as np
import random
import pandas as pd
import json
import pdb
import os


def area_detection_test(directory_path, result_path, model_path):

    os.makedirs(result_path, exist_ok=True)

    clear_result_path(result_path)

    dataset_original = FCGPairDataset_test(directory_path)

    input_dim = dataset_original.pairs[0]['candidate'].x.size(1)

    model_test = SiameseGCNConv(input_dim)

    model_test.load_state_dict(torch.load(model_path))

    area_test(model_test, dataset_original, result_path)

def area_test(model, test_loader, result_path):
    device = "cuda"
    model.eval()
    model.to(device)

    results = pd.DataFrame(columns=['file_name', 'func_name', 'output'])

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing", unit="batch"):
            file_name = batch[2]
            func_name = batch[3]
            candidate = batch[0].to(device)
            target = batch[1].to(device)

            output = model(candidate, target)
            score = output[0].item()
            pred = 1 if score >= 0.5 else 0

            func_a, func_b = func_name.split('---')
            label = 1 if func_a == func_b else 0

            all_preds.append(pred)
            all_labels.append(label)

            new_row = pd.DataFrame({
                'file_name': [file_name[0]],
                'func_name': [func_name[0]],
                'output': [score]
            })
            results = pd.concat([results, new_row], ignore_index=True)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_names = results['file_name'].unique()

    for file_name in file_names:
        file_df = results[results['file_name'] == file_name]

        csv_file_name = file_name.replace(".json", ".csv")
        file_path = os.path.join(result_path, csv_file_name)
        file_df.to_csv(file_path, index=False)
    print(f"Results have been saved to {result_path}")

def clear_result_path(result_path):
    """Clears the contents of the result_path directory."""
    if os.path.exists(result_path):
        for filename in os.listdir(result_path):
            file_path = os.path.join(result_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    else:
        print(f"Directory {result_path} does not exist.")


