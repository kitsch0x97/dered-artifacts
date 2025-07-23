import os
import json
import pickle
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

keys_to_remove = ['.open', '.exit', '.log2', '.feof', '.__errno_location', '.memcpy', '.__stack_chk_fail', '.__xstat',
                  '.fwrite', '.fdopen', '.strcmp', '.utime', '.__memset_chk', '.memset', '.ferror', '.chown', '.fclose',
                  '.strncmp', '.unlink', '.malloc', '.init_proc', '.fseek', '.chmod', '.strcpy', '.memmove', '.ftell',
                  '.__fprintf_chk', '.__gmon_start__', '.__libc_start_main', '.strerror', '.strlen', '.fopen',
                  '.strchr', '.isatty', '.strrchr', '.fread', '.free']

def reuse_detection(target_binary_path, candidate_binary_path, json_path, target_fcg_path, candidate_fcg_path, target_exp_path, candidate_exp_path, result_path, gt_path,
                    alpha, num_walks, mode, threshold):

    result_dict = {}

    diff_list = []
    os.makedirs(result_path, exist_ok=True)

    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]


    with open(gt_path, 'r') as file:
        result_gt = json.load(file)

    total_df = pd.DataFrame()
    for json_file in json_files:
        single_df = process_single_file(
            json_file, json_path, target_binary_path, candidate_binary_path, result_gt,
            target_fcg_path, candidate_fcg_path, target_exp_path, candidate_exp_path,
            alpha, num_walks, mode, result_dict, diff_list, threshold
        )
        if not single_df.empty:
            total_df = pd.concat([total_df, single_df], axis=0, ignore_index=True)

    total_df.to_csv(os.path.join(result_path,'total_df.csv'), index=False)

    total_df = pd.read_csv(os.path.join(result_path, 'total_df.csv'))
    trained_model = train_random_forest(total_df, result_path)

def process_single_file(json_file, json_path, target_binary_path, candidate_binary_path, result_gt, target_fcg_path,
            candidate_fcg_path, target_exp_path, candidate_exp_path, alpha, num_walks, mode, result_dict, diff_list, threshold):
    file_path = os.path.join(json_path, json_file)
    data = load_json(file_path)

    if len(data.keys()) < 1:
        return pd.DataFrame()

    target_name = os.path.basename(file_path).replace('.json', '').split('---')[0]
    candidate_name = os.path.basename(file_path).replace('.json', '').split('---')[1]

    (
        target_ef_value_list, candidate_ef_value_list,
        target_pr_list, candidate_pr_list,
        target_area_size_list, candidate_area_size_list,
        func_pair_list, target_func_size,
        candidate_func_size, target_ef_size,
        candidate_ef_size
    ) = process_target_candidate_v2(
        data, target_name, candidate_name,
        target_fcg_path, candidate_fcg_path,
        target_exp_path, candidate_exp_path,
        alpha, num_walks, mode, result_dict,
        diff_list, threshold
    )

    file_pair = f"{target_name}---{candidate_name}"
    data_dict = {
        'target_ef_value': target_ef_value_list,
        'candidate_ef_value': candidate_ef_value_list,
        'target_pr_value': target_pr_list,
        'candidate_pr_value': candidate_pr_list,
        'file_pair': file_pair,
        'func_pair': func_pair_list

    }

    df = pd.DataFrame(data_dict)

    df['label'] = 0
    if target_name in result_gt and candidate_name in result_gt[target_name]:
        df['label'] = 1

    return df

def process_target_candidate_v2(
    data, target_name, candidate_name, target_fcg_path, candidate_fcg_path,
    target_exp_path, candidate_exp_path, alpha, num_walks, mode,
    result_dict, diff_list, threshold
):

    target_path = os.path.join(target_fcg_path, target_name + "_fcg.pkl")
    candidate_path = os.path.join(candidate_fcg_path, candidate_name + "_fcg.pkl")
    target_export_path = os.path.join(target_exp_path, target_name + "_export.json")
    candidate_export_path = os.path.join(candidate_exp_path, candidate_name + "_export.json")

    target_data = load_pickle(target_path)
    candidate_data = load_pickle(candidate_path)

    if target_data is None or candidate_data is None:
        return

    target_pagerank = nx.pagerank(target_data)
    candidate_pagerank = nx.pagerank(candidate_data)

    target_ef_data = filter_top_functions(load_json(target_export_path), target_pagerank, fraction=1)
    candidate_ef_data = filter_top_functions(load_json(candidate_export_path), candidate_pagerank, fraction=1)

    target_func_size = len(target_data.nodes())
    candidate_func_size = len(candidate_data.nodes())

    target_ef_list = list(set(target_ef_data.keys()))
    candidate_ef_list = list(set(candidate_ef_data.keys()))

    target_ef_size = len(target_ef_list)
    candidate_ef_size = len(candidate_ef_list)

    target_ef_value_list, candidate_ef_value_list, target_pr_list, candidate_pr_list, target_area_size_list, candidate_area_size_list, func_pair_list\
        = compute_tpl_diff_all_v2(data, target_name, candidate_name, target_data, candidate_data, target_pagerank, candidate_pagerank, target_ef_list, candidate_ef_list, alpha, num_walks, threshold)

    return target_ef_value_list, candidate_ef_value_list, target_pr_list, candidate_pr_list, target_area_size_list, candidate_area_size_list, func_pair_list, target_func_size, candidate_func_size, target_ef_size, candidate_ef_size

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return None

def filter_top_functions(ef_data, pagerank, fraction=0.1):
    sorted_functions = sorted(
        ef_data.keys(),
        key=lambda func: pagerank.get(func, 0),
        reverse=True
    )
    top_count = max(1, int(len(sorted_functions) * fraction))
    top_functions = sorted_functions[:top_count]

    filtered_ef_data = {func: ef_data[func] for func in top_functions}
    return filtered_ef_data

def compute_tpl_diff_all_v2(data, target_name, candidate_name, target_data, candidate_data, target_pagerank, candidate_pagerank, target_ef_list,
                                            candidate_ef_list, alpha, num_walks, threshold):

    target_ef_value_list = []
    candidate_ef_value_list = []
    target_pr_list = []
    candidate_pr_list = []
    target_area_size_list = []
    candidate_area_size_list = []
    func_pair_list = []

    for key in data.keys():

        if data[key]['result'] == '0':
            continue

        target_area_func, candidate_area_func = data[key]['target_fcg']['nodes'], data[key]['candidate_fcg']['nodes']
        target_area_size = len(target_area_func)
        candidate_area_size = len(candidate_area_func)

        target_ef = compute_external_link_feature(target_data, target_area_func, target_ef_list)
        candidate_ef = compute_external_link_feature(candidate_data, candidate_area_func, candidate_ef_list)

        target_pr = compute_internal_link_feature(target_data, target_area_func, target_pagerank)
        candidate_pr = compute_internal_link_feature(candidate_data, candidate_area_func, candidate_pagerank)

        if target_ef is None or candidate_ef is None or target_pr is None or candidate_pr is None or target_area_size is None or candidate_area_size is None:
            continue

        target_ef_value_list.append(target_ef)
        candidate_ef_value_list.append(candidate_ef)
        target_pr_list.append(target_pr)
        candidate_pr_list.append(candidate_pr)
        target_area_size_list.append(target_area_size)
        candidate_area_size_list.append(candidate_area_size)
        func_pair_list.append(key)
    return target_ef_value_list, candidate_ef_value_list, target_pr_list, candidate_pr_list, target_area_size_list, candidate_area_size_list, func_pair_list

def compute_external_link_feature(g: nx.DiGraph, g_fa: set, E: list) -> float:

    phi = {node: 1.0 if node in g_fa else 0.0 for node in g.nodes()}

    non_target_nodes = [node for node in g.nodes() if node not in g_fa]

    max_iter = 100
    tolerance = 1e-6
    for _ in range(max_iter):
        delta = 0.0
        new_phi = phi.copy()

        for node in non_target_nodes:
            successors = list(g.successors(node))
            out_degree = len(successors)

            if out_degree == 0:
                new_phi[node] = 0.0
            else:
                prob_sum = sum(phi[s] for s in successors) / out_degree
                new_phi[node] = prob_sum

            delta = max(delta, abs(new_phi[node] - phi[node]))

        phi = new_phi
        if delta < tolerance:
            break

    valid_E = [e for e in E if e in phi]
    if not E:
        return 0.0

    total = sum(phi.get(e, 0.0) for e in E)
    F_ext = total / len(E)

    return F_ext

def train_random_forest(total_df, result_path):
    if total_df.empty:
        raise ValueError("total_df 是空的，无法训练模型！")

    total_df_with_predictions = total_df.copy()

    X = total_df.iloc[:, :-3]
    y = total_df.iloc[:, -1]
    file_pairs = total_df.iloc[:, -3]
    func_pairs = total_df.iloc[:, -2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    test_indices = X_test.index.tolist()

    results = pd.DataFrame({
        "true_label": y_test,
        "pred_label": y_pred,
        "file_pair": file_pairs.iloc[test_indices].values,
        "func_pair": func_pairs.iloc[test_indices].values
    }, index=test_indices)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    fp_mask = (results["true_label"] == 0) & (results["pred_label"] == 1)
    fn_mask = (results["true_label"] == 1) & (results["pred_label"] == 0)

    fp_pairs = results[fp_mask][["func_pair"]].values.tolist()
    fn_pairs = results[fn_mask][["func_pair"]].values.tolist()

    print(f"误报数量: {len(fp_pairs)}")
    print(f"误报信息: ")
    for fp in fp_pairs:
        print(fp)

    print(f"漏报数量: {len(fn_pairs)}")
    print(f"漏报信息:")
    for fn in fn_pairs:
        print(fn)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n模型准确率: {accuracy:.4f}")
    print("\n分类报告:\n", classification_report(y_test, y_pred))

    total_df_with_predictions['predicted_label'] = rf_model.predict(X)  # 使用训练好的模型预测全部样本
    total_df_with_predictions.to_csv(os.path.join(result_path, 'total_df_with_predictions.csv'), index=False)

    save_dir = "/data/csj/Faker/code/libam/data/model/"
    model_path = os.path.join(save_dir, "rf_model.joblib")
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)


    return {
        "model": rf_model,
        "false_positives": fp_pairs,
        "false_negatives": fn_pairs,
    }

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file: {file_path}")
        print(f"Details: {e}")
        return None

def compute_internal_link_feature(
        g: nx.DiGraph,
        target_area_func: list,
        target_pagerank: dict,
        epsilon: float = 1e-6
) -> float:

    target_set = set(target_area_func)
    F_int = 0.0

    for func in target_area_func:
        E_int = 0
        E_ext = 0

        neighbors = list(g.predecessors(func)) + list(g.successors(func))

        for neighbor in neighbors:
            if neighbor in target_set:
                E_int += 1
            else:
                E_ext += 1

        total_edges = E_int + E_ext + epsilon
        omega = E_ext / total_edges if total_edges != 0 else 0.0

        pr = target_pagerank.get(func, 0.0)
        F_int += omega * pr

    return F_int









