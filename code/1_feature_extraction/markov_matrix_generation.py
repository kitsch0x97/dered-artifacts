#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import pdb
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from instruction_sets.all_instruction_sets_deepseek import (
    arith_set as arith_set_deepseek,
    data_transfer_set as data_transfer_set_deepseek,
    cmp_set as cmp_set_deepseek,
    logic_set as logic_set_deepseek,
    shift_set as shift_set_deepseek,
    unconditional_set as unconditional_set_deepseek,
    conditional_set as conditional_set_deepseek,
    memory_management_set as memory_management_set_deepseek,
    processor_state_set as processor_state_set_deepseek,
    synchronization_set as synchronization_set_deepseek,
    vector_management_set as vector_management_set_deepseek
)

from instruction_sets.all_instruction_sets_gemini import (
    arith_set as arith_set_gemini,
    data_transfer_set as data_transfer_set_gemini,
    cmp_set as cmp_set_gemini,
    logic_set as logic_set_gemini,
    shift_set as shift_set_gemini,
    unconditional_set as unconditional_set_gemini,
    conditional_set as conditional_set_gemini,
    memory_management_set as memory_management_set_gemini,
    processor_state_set as processor_state_set_gemini,
    synchronization_set as synchronization_set_gemini,
    vector_management_set as vector_management_set_gemini
)

from instruction_sets.all_instruction_sets_gpt import (
    arith_set as arith_set_gpt,
    data_transfer_set as data_transfer_set_gpt,
    cmp_set as cmp_set_gpt,
    logic_set as logic_set_gpt,
    shift_set as shift_set_gpt,
    unconditional_set as unconditional_set_gpt,
    conditional_set as conditional_set_gpt,
    memory_management_set as memory_management_set_gpt,
    processor_state_set as processor_state_set_gpt,
    synchronization_set as synchronization_set_gpt,
    vector_management_set as vector_management_set_gpt
)

from instruction_sets.all_instruction_sets_final import (
    arith_set as arith_set_final,
    data_transfer_set as data_transfer_set_final,
    cmp_set as cmp_set_final,
    logic_set as logic_set_final,
    shift_set as shift_set_final,
    unconditional_set as unconditional_set_final,
    conditional_set as conditional_set_final,
    memory_management_set as memory_management_set_final,
    processor_state_set as processor_state_set_final,
    synchronization_set as synchronization_set_final,
    vector_management_set as vector_management_set_final
)

def generateMarkovMatrix(cfg_path, save_path):
    if not os.path.exists(cfg_path):
        print(f"Path {cfg_path} does not exist.")
        return None

    all_files = []
    for root, _, files in os.walk(cfg_path):
        for file_name in files:
            if file_name.endswith(".json"):
                all_files.append(os.path.join(root, file_name))

    updateInstructionSet()

    multiFileProcess(all_files, save_path)

def updateInstructionSet():

    arith_set = set(arith_set_deepseek & arith_set_gemini & arith_set_gpt)
    data_transfer_set = set(data_transfer_set_deepseek & data_transfer_set_gemini & data_transfer_set_gpt)
    cmp_set = set(cmp_set_deepseek & cmp_set_gemini & cmp_set_gpt)
    logic_set = set(logic_set_deepseek & logic_set_gemini & logic_set_gpt)
    shift_set = set(shift_set_deepseek & shift_set_gemini & shift_set_gpt)
    unconditional_set = set(unconditional_set_deepseek & unconditional_set_gemini & unconditional_set_gpt)
    conditional_set = set(conditional_set_deepseek & conditional_set_gemini & conditional_set_gpt)
    memory_management_set = set(memory_management_set_deepseek & memory_management_set_gemini & memory_management_set_gpt)
    processor_state_set = set(processor_state_set_deepseek & processor_state_set_gemini & processor_state_set_gpt)
    synchronization_set = set(synchronization_set_deepseek & synchronization_set_gemini & synchronization_set_gpt)
    vector_management_set = set(vector_management_set_deepseek & vector_management_set_gemini & vector_management_set_gpt)

    module_path = os.path.abspath("/data/csj/DeRed/code/1_feature_extraction/instruction_sets/all_instruction_sets_final.py")

    file_content = f"""# Auto-generated instruction sets
arith_set = {repr(arith_set)}
data_transfer_set = {repr(data_transfer_set)}
cmp_set = {repr(cmp_set)}
logic_set = {repr(logic_set)}
shift_set = {repr(shift_set)}
unconditional_set = {repr(unconditional_set)}
conditional_set = {repr(conditional_set)}
memory_management_set = {repr(memory_management_set)}
processor_state_set = {repr(processor_state_set)}
synchronization_set = {repr(synchronization_set)}
vector_management_set = {repr(vector_management_set)}
"""
    try:
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        print(f"\nSuccessfully updated {module_path}")
    except Exception as e:
        print(f"Error saving instruction sets: {str(e)}")
        raise

def multiFileProcess(all_files, save_path):

    max_workers = min(30, len(all_files)) 
    timeout = 600  

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                singleFileProcess,
                file_path=file_path,
                save_path=save_path,
            ): file_path for file_path in all_files
        }

        progress_bar = tqdm(
            as_completed(future_to_file),
            total=len(all_files),
            desc="Processing Files",
            unit="file",
            dynamic_ncols=True
        )

        for future in progress_bar:
            file_path = future_to_file[future]
            try:
                future.result(timeout=timeout) 
                progress_bar.set_postfix(
                    file=os.path.basename(file_path)[:15],  
                    status="OK"
                )
            except FileNotFoundError as e:
                tqdm.write(f"\nFile not found: {e.filename}")
            except json.JSONDecodeError as e:
                tqdm.write(f"\nJSON parsing error [{file_path}]: {str(e)}")
            except Exception as e:
                tqdm.write(f"\nUnknown error [{file_path}]: {str(e)}")

            finally:
                progress_bar.update(1)

def singleFileProcess(file_path, save_path):
    try:
        with open(file_path, "r") as json_file:
            functions_info = json.load(json_file)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return

    file_string = re.search(r'([^/]+(?:_[^/]+)*)_cfg\.json', file_path).group(1)
    output_dir = os.path.join(save_path, file_string)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"{file_string}_markov.csv")
    all_features_df = pd.DataFrame()

    for func_name, info in functions_info.items():
        prob_matrix = singleFunctionProcess(info)

        combined_features = prob_matrix.flatten().tolist()

        markov_features_df = pd.DataFrame(
            [combined_features],
            columns=[f"feature_{i}" for i in range(len(combined_features))]
        )

        markov_features_df["function_name"] = func_name
        markov_features_df["file_name"] = file_string 

        all_features_df = pd.concat(
            [all_features_df, markov_features_df],
            axis=0,
            ignore_index=True
        )

    all_features_df.to_csv(output_file, index=False)

def singleFunctionProcess(info):

    category_map = [
        (arith_set_final, "1"), (data_transfer_set_final, "2"), (cmp_set_final, "3"),
        (logic_set_final, "4"), (shift_set_final, "5"), (unconditional_set_final, "6"),
        (conditional_set_final, "7"), (memory_management_set_final, "8"), (processor_state_set_final, "9"),
        (synchronization_set_final, "10"), (vector_management_set_final, "11"),
    ]

    basic_blocks = info.get("basic_blocks", [])

    positive_count, negative_count = 0, 0
    matched_instr = []

    for block in basic_blocks:
        instructions = block.get("instructions", [])
        for instr in instructions:
            instr_upper = instr.upper()
            matched = False
            for instr_set, label in category_map:
                if instr_upper in instr_set:
                    matched_instr.append(label)
                    matched = True
                    break

            if matched:
                positive_count = positive_count + 1
            else:
                negative_count = negative_count + 1
                matched_instr.append("8")

    matrix = build_prob_matrix(matched_instr, len(category_map))

    return matrix

def build_prob_matrix(matched_instr, max_state):

    transition_matrix = np.zeros((max_state, max_state), dtype=int)

    if len(matched_instr) >= 2:
        for i in range(len(matched_instr) - 1):
            current = int(matched_instr[i]) - 1  
            next_state = int(matched_instr[i + 1]) - 1

            if 0 <= current <= max_state and 0 <= next_state <= max_state:
                transition_matrix[current][next_state] += 1

    row_sums = np.sum(transition_matrix, axis=1)
    non_zero_mask = row_sums > 0
    transition_matrix = transition_matrix.astype(float) 
    transition_matrix[non_zero_mask] = transition_matrix[non_zero_mask] / row_sums[non_zero_mask][:, np.newaxis]

    return transition_matrix