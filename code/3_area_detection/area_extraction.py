
import ast
import shutil
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pickle
import os
import networkx as nx
from collections import deque

global_max_depth = 5

# 初始化缓存字典
file_cache = {}
target_fcg_cache = {}
candidate_fcg_cache = {}


def area_extraction_train(gt_path, feature_path, target_path, candidate_path, output_dir):

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_dict = json.load(f)

    file_pairs = gt_dict.keys()
    target_fcg_path_set = set()
    candidate_fcg_path_set = set()

    clear_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for file_pair in file_pairs:

        target_file, candidate_file = file_pair.split('---')

        target_fcg_path = os.path.join(target_path, f"{target_file}_fcg.pkl")
        candidate_fcg_path = os.path.join(candidate_path, f"{candidate_file}_fcg.pkl")

        target_fcg_path_set.add(target_fcg_path)
        candidate_fcg_path_set.add(candidate_fcg_path)

    for target_fcg_path in target_fcg_path_set:
        if os.path.exists(target_fcg_path):
            with open(target_fcg_path, 'rb') as f:
                target_fcg_cache[target_fcg_path] = pickle.load(f)

    for candidate_fcg_path in candidate_fcg_path_set:
        if os.path.exists(candidate_fcg_path):
            with open(candidate_fcg_path, 'rb') as f:
                candidate_fcg_cache[candidate_fcg_path] = pickle.load(f)


    multi_threaded_processing_train(feature_path, file_pairs, gt_dict, target_path, candidate_path, output_dir)

def clear_output_dir(output_dir):

    # 如果目录存在，删除该目录及其所有内容
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 递归删除整个目录及其中的所有文件和子目录
        print(f"目录 {output_dir} 的内容已被删除")

    # 重新创建空目录
    os.makedirs(output_dir)
    print(f"空目录 {output_dir} 已创建")

def multi_threaded_processing_train(feature_path, file_pairs, gt_dict, target_path, candidate_path, output_dir):

    max_workers = min(30, len(file_pairs))  
    timeout = 600  

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_dict = {
            executor.submit(
                process_single_csv_train,
                feature_path,
                file_pair,
                gt_dict[file_pair],  
                target_path,
                candidate_path,
                output_dir
            ): file_pair for file_pair in file_pairs
        }

        progress_bar = tqdm(
            as_completed(future_dict),
            total=len(file_pairs),
            desc="Processing file pairs",
            unit="pair",
            dynamic_ncols=True
        )

        for future in progress_bar:
            file_pair = future_dict[future]
            try:
                future.result(timeout=timeout)  
                progress_bar.set_postfix(
                    pair=str(file_pair)[:15],  
                    status="OK"
                )
            except pd.errors.EmptyDataError:
                pass
            except FileNotFoundError as e:
                pass
            except TimeoutError:
                pass
            except Exception as e:
                pass
            finally:
                progress_bar.update(1)

def process_single_csv_train(feature_path, file_pair, anchor_func_list, target_path, candidate_path, output_dir):
    subgraph_dic = {
        "file_pairs": [],
        "function_pairs": [],
        "all_target_subgraphs": [],
        "all_candidate_subgraphs": [],
        "target_feature": [],
        "candidate_feature": [],
        "label": []

    }

    target_file, candidate_file = file_pair.split('---')

    anchor_path = os.path.join(feature_path, f"{file_pair}_dataset.csv")

    if os.path.exists(anchor_path):
        pass

    anchor_df = pd.read_csv(anchor_path)
    function_pairs = anchor_df['function_pair'].tolist()
    same_pairs = []
    different_pairs = []

    for pair in function_pairs:
        func1, func2 = pair.split('---')

        if func1 == func2:
            same_pairs.append(pair)
        else:
            different_pairs.append(pair)

    target_anchor_df = anchor_df.drop(columns=["embed2", "file_pair", "label"])
    candidate_anchor_df = anchor_df.drop(columns=["embed1", "file_pair", "label"])
    target_anchor_df['function_pair'] = target_anchor_df['function_pair'].str.split('---').str[0]
    candidate_anchor_df['function_pair'] = candidate_anchor_df['function_pair'].str.split('---').str[-1]
    target_graph = load_graph_safely(target_fcg_cache, target_path, target_file)
    if target_graph is None:
        return target_graph

    candidate_graph = load_graph_safely(candidate_fcg_cache, candidate_path, candidate_file)
    if candidate_graph is None:
        return candidate_graph

    target_nodes = set(target_graph.nodes())
    candidate_nodes = set(candidate_graph.nodes())

    for same_pair in same_pairs:
        target_func, candidate_func = same_pair.split('---')
        if target_func in target_nodes and candidate_func in candidate_nodes:

            target_depth  = calculate_depth(target_graph, target_func)
            candidate_depth  = calculate_depth(candidate_graph, candidate_func)

            common_depth = min(target_depth, candidate_depth)

            if common_depth * 2 < 1:
                continue

            target_subgraph = extract_subgraph(target_graph, target_func, common_depth)
            candidate_subgraph = extract_subgraph(candidate_graph, candidate_func, common_depth)

            subgraph_dic["file_pairs"].append(file_pair)
            subgraph_dic["function_pairs"].append(f"{target_func}---{candidate_func}")
            subgraph_dic["all_target_subgraphs"].append(target_subgraph)
            subgraph_dic["all_candidate_subgraphs"].append(candidate_subgraph)

            target_feature = generate_feature_train(target_subgraph, target_anchor_df)
            candidate_feature = generate_feature_train(candidate_subgraph, candidate_anchor_df)

            subgraph_dic["target_feature"].append(target_feature)
            subgraph_dic["candidate_feature"].append(candidate_feature)
            subgraph_dic["label"].append('1')

    for same_pair in different_pairs:
        target_func, candidate_func = same_pair.split('---')
        if target_func in target_nodes and candidate_func in candidate_nodes:

            target_depth  = calculate_depth(target_graph, target_func)
            candidate_depth  = calculate_depth(candidate_graph, candidate_func)

            common_depth = min(target_depth, candidate_depth)

            if common_depth * 2 < 1:
                continue

            target_subgraph = extract_subgraph(target_graph, target_func, common_depth)
            candidate_subgraph = extract_subgraph(candidate_graph, candidate_func, common_depth)

            subgraph_dic["file_pairs"].append(file_pair)
            subgraph_dic["function_pairs"].append(f"{target_func}---{candidate_func}")
            subgraph_dic["all_target_subgraphs"].append(target_subgraph)
            subgraph_dic["all_candidate_subgraphs"].append(candidate_subgraph)

            target_feature = generate_feature_train(target_subgraph, target_anchor_df)
            candidate_feature = generate_feature_train(candidate_subgraph, candidate_anchor_df)

            subgraph_dic["target_feature"].append(target_feature)
            subgraph_dic["candidate_feature"].append(candidate_feature)
            subgraph_dic["label"].append('0')

    if len(subgraph_dic['file_pairs'])==0:
        return

    graph_dict = {}
    for graph_num, (function_pair, target_subgraph, candidate_subgraph, target_feature, candidate_feature, label) \
        in enumerate(zip(subgraph_dic["function_pairs"], subgraph_dic["all_target_subgraphs"],
                subgraph_dic["all_candidate_subgraphs"], subgraph_dic["target_feature"],
                subgraph_dic["candidate_feature"], subgraph_dic["label"]),
            start=1
    ):
        target_func, candidate_func = function_pair.split('---')
        target_graph_dict = {}
        candidate_graph_dict = {}
        save_subgraph_test(target_graph_dict, target_subgraph, target_func, 'target_fcg', target_feature)
        save_subgraph_test(candidate_graph_dict, candidate_subgraph, candidate_func, 'candidate_fcg', candidate_feature)
        graph_dict[function_pair] = {  # 将结果绑定到函数对
            "target_fcg": target_graph_dict["target_fcg"],
            "candidate_fcg": candidate_graph_dict["candidate_fcg"],
            "result": label
        }

    if len(graph_dict.keys())>0:
        with open(os.path.join(output_dir, f'{file_pair}.json'), 'w') as file:
            json.dump(graph_dict, file, indent=4)

def load_graph_safely(graph_cache, path, filename):
    try:
        file_path = os.path.join(path, f"{filename}_fcg.pkl")
        return graph_cache[file_path]
    except KeyError:
        return None
    except Exception as e:
        return None

def calculate_depth(target_subgraph, target_func):
    queue = deque([(target_func, 0)])
    visited = set([target_func])
    max_depth = 0

    while queue:
        current_node, depth = queue.popleft()
        max_depth = max(max_depth, depth)

        for neighbor in target_subgraph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return max_depth

def extract_subgraph(graph,start_function, common_depth):

    function_region_nodes = get_children_list_v2(start_function, graph, walked_map=None, current_depth=0, max_depth=common_depth)

    function_region_nodes.append(start_function)

    subgraph = graph.subgraph(function_region_nodes).copy()
    return subgraph

def get_children_list_v2(start_node, graph, max_depth, walked_map=None, current_depth=0):

    if walked_map is None:
        walked_map = set()
    stack = [(start_node, current_depth, walked_map.copy())]
    all_children = set()

    while stack:
        node, depth, path = stack.pop()
        if depth > max_depth or node in path:
            continue
        path.add(node)
        all_children.add(node)
        children = get_child_node_v2(node, graph)
        for child in children:
            if child not in path and depth < max_depth:
                stack.append((child, depth + 1, path.copy()))
    return list(all_children - {start_node})  
def generate_feature_train(subgraph, df):

    feature_list = []
    target_nodes = subgraph.nodes()
    feature_len = len(ast.literal_eval(df.iloc[0][0]))
    for node in target_nodes:

        df_func = df[df['function_pair'] == node]

        if df_func.shape[0] > 0:
            feature = ast.literal_eval(df_func.iloc[0][0])
        else:
            feature = [0.0] * feature_len


        feature_list.append(feature)
    return json.dumps(feature_list)

def get_child_node_v2(start_node, graph):
    return list(graph.successors(start_node))

def save_subgraph_test(integrated_graph_dict, graph, function, flag, feature):

    integrated_graph = integrate_call_graph_test(graph, function, feature)
    integrated_graph_dict[flag] = integrated_graph

def integrate_call_graph_test(call_graph, start_function, feature):

    call_graph = generate_call_graph(call_graph, start_function)

    integrated_graph = {'nodes': [start_function], 'edges': []}

    for function, details in call_graph.items():
        integrated_graph['nodes'] += details['nodes']
        integrated_graph['edges'] += details['edges']

    integrated_graph['nodes'] = list(set(integrated_graph['nodes']))
    integrated_graph['edges'] = list(set(integrated_graph['edges']))
    integrated_graph['feature'] = feature
    return integrated_graph

def generate_call_graph(graph, function, visited=None, call_graph=None):

    if visited is None:
        visited = set()

    if call_graph is None:
        call_graph = {}

    if function in visited:
        return call_graph

    visited.add(function)

    neighbors = list(graph.neighbors(function))

    call_graph[function] = {
        "nodes": neighbors,
        "edges": [(function, neighbor) for neighbor in neighbors]
    }

    for neighbor in neighbors:
        generate_call_graph(graph, neighbor, visited, call_graph)

    return call_graph










