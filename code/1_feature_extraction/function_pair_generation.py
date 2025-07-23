#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pdb
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

def functionPairGeneration(statistic_path_1, statistic_path_2, save_path):
    binary_function_features_1 = load_data(statistic_path_1)
    binary_function_features_2 = load_data(statistic_path_2)

    file_names_1 = [item['file_name'][0] for item in binary_function_features_1]
    file_names_2 = [item['file_name'][0] for item in binary_function_features_2]

    combined_binary_function_features = binary_function_features_1 + binary_function_features_2

    X_target, X_candidate, X, y, file_pairs, function_pairs, binary_dataset = create_dataset(combined_binary_function_features, file_names_1, file_names_2, 3)

    print(f'complete data process. number of positive: {np.sum(y)}, number of negative: {len(y)-np.sum(y)}')

    X_target_df = pd.DataFrame(X_target, columns=combined_binary_function_features[0].columns[:-2].tolist())
    X_candidate_df = pd.DataFrame(X_candidate, columns=combined_binary_function_features[0].columns[:-2].tolist())

    X_target_df['label'] = y
    X_target_df['file_pair'] = file_pairs
    X_target_df['function_pair'] = function_pairs

    X_candidate_df['label'] = y
    X_candidate_df['file_pair'] = file_pairs
    X_candidate_df['function_pair'] = function_pairs

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X_target_df.to_csv(os.path.join(save_path, 'target_dataset.csv'), index=False)
    X_candidate_df.to_csv(os.path.join(save_path, 'candidate_dataset.csv'), index=False)

def load_data(statistic_path):
    if not os.path.exists(statistic_path):
        print(f"Path {statistic_path} does not exist.")
        return None

    binary_function_features = []

    csv_files = [
        os.path.join(root, file_name)
        for root, _, files in os.walk(statistic_path)
        for file_name in files if file_name.endswith(".csv")
    ]

    for file_path in tqdm(csv_files, desc="Loading and processing CSV files"):
        try:
            df = pd.read_csv(file_path)
            df = df.dropna()
            df = df.drop_duplicates()

            for col in df.columns[:-2]:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].astype(float)
                    except ValueError:
                        pass

            for col in df.columns[:-2]:

                non_numeric_rows = df[pd.to_numeric(df[col], errors='coerce').isna()]
                if not non_numeric_rows.empty:
                    print(f"Column '{col}' contains non-numeric data:")
                    print(non_numeric_rows[[col]])

            binary_function_features.append(df)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            print(df[df.columns[:-2]])
            continue

    return binary_function_features

def create_dataset(binary_function_features, file_names_1, file_names_2, neg_index):
    merged_df = pd.concat(binary_function_features)
    preserved_columns = merged_df.columns.tolist()  
    function_name_map = defaultdict(list)

    for func_name, group in merged_df.groupby('function_name'):
        function_name_map[func_name] = group[preserved_columns].values.tolist()

    binary_dataset = []

    for func, rows in tqdm(function_name_map.items(), desc="Generating pairs..."):
        file_dict = defaultdict(list)
        for row in rows:
            file_name = row[preserved_columns.index('file_name')]  
            file_dict[file_name].append(row)

        files = list(file_dict.keys())
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                for r1 in file_dict[files[i]]:
                    for r2 in file_dict[files[j]]:
                        binary_dataset.append((r1, r2, 1))

    function_list = list(function_name_map.items())
    neg_needed = len(binary_dataset) * neg_index
    neg_count = 0

    if len(function_list) >= 2:
        with tqdm(total=neg_needed, desc="Generating negatives") as pbar:
            while neg_count < neg_needed:
                (func1, rows1), (func2, rows2) = random.sample(function_list, 2)
                if func1 != func2:
                    binary_dataset.append((
                        random.choice(rows1),
                        random.choice(rows2),
                        0
                    ))
                    neg_count += 1
                    pbar.update(1)

    random.shuffle(binary_dataset)
    X_target, X_candidate, X, y, file_pairs, function_pairs = create_feature_pairs_fast(binary_dataset, file_names_1, file_names_2)

    return X_target, X_candidate, X, y, file_pairs, function_pairs, binary_dataset

def create_feature_pairs_fast(binary_dataset, file_names_1, file_names_2):

    X_target, X_candidate, X, y, file_pairs, function_pairs = [], [], [], [], [], []

    for batch in tqdm(binary_dataset, desc="binary_dataset"):
        row1, row2, label = batch
        if row1[-1] in file_names_1 and row2[-1] in file_names_2:

            feature1 = np.array(row1[:-2], dtype=np.float32)
            feature2 = np.array(row2[:-2], dtype=np.float32)

            diff = feature1 - feature2

            file_pair = f"{row1[-1]}---{row2[-1]}"
            function_pair = f"{row1[-2]}---{row2[-2]}"

            X_target.append(feature1)
            X_candidate.append(feature2)
            X.append(diff)
            y.append(label)
            file_pairs.append(file_pair)
            function_pairs.append(function_pair)

    X_target = np.stack(X_target)
    X_candidate = np.stack(X_candidate)
    X = np.stack(X)
    y = np.array(y)
    return X_target, X_candidate, X, y, file_pairs, function_pairs

def process_target_func(target_func, binary_function_features_2):
    local_target = []
    local_candidate = []

    target_func_name = target_func['function_name']
    target_file_name = target_func['file_name']
    target_feature = target_func[:-2].values

    for _, candidate_func in binary_function_features_2.iterrows():
        candidate_func_name = candidate_func['function_name']
        candidate_file_name = candidate_func['file_name']
        candidate_feature = candidate_func[:-2].values

        if target_file_name == candidate_file_name:
            continue

        func_pair = f"{target_func_name}---{candidate_func_name}"
        file_pair = f"{target_file_name}---{candidate_file_name}"

        local_target.append(np.concatenate([target_feature, [file_pair, func_pair]]))
        local_candidate.append(np.concatenate([candidate_feature, [file_pair, func_pair]]))

    return local_target, local_candidate

def combined_df_generation(target_func, candidate_func):

    target_func['_temp_key'] = 1
    candidate_func['_temp_key'] = 1

    combined_df = target_func.merge(candidate_func, on='_temp_key', suffixes=('_target', '_candidate'))
    combined_df["function_name"] = combined_df["function_name_target"] + "---" + combined_df["function_name_candidate"]
    combined_df["file_name"] = combined_df["file_name_target"] + "---" + combined_df["file_name_candidate"]
    combined_df.drop(
        columns=["file_name_target", "file_name_candidate", "function_name_target", "function_name_candidate"],
        inplace=True)
    combined_df.drop('_temp_key', axis=1, inplace=True)
    pdb.set_trace()
    return combined_df


