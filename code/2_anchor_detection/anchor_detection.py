import os
import time
import json
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def anchor_detection_v2(dataset_dir, model_path, savePath):
    os.makedirs(savePath, exist_ok=True)
    print('anchor_detection_v2---create savePath')

    target_file_path = os.path.join(dataset_dir, 'target_dataset.csv')
    candidate_file_path = os.path.join(dataset_dir, 'candidate_dataset.csv')
    target_df = pd.read_csv(target_file_path)
    candidate_df = pd.read_csv(candidate_file_path)
    print('anchor_detection_v2---load datasets')

    target_features = target_df.iloc[:, :-3].values
    candidate_features = candidate_df.iloc[:, :-3].values
    labels = target_df.iloc[:, -3].values

    test_loader = create_siamese_dataloader(
        target_features, candidate_features, labels,
        batch_size=512, shuffle=False
    )

    model = load_trained_model(model_path, input_dim=target_features.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y_pred, y_true = [], []
    total_time = 0.0
    chunk_size = 10000
    chunk_idx = 0
    chunk_results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            start_time = time.time()
            input1, input2, labels_batch = batch
            input1, input2 = input1.to(device), input2.to(device)

            outputs, embed1, embed2 = model(input1, input2)
            batch_pred = (outputs.cpu() > 0.5).int().numpy()

            y_pred.extend(batch_pred)
            y_true.extend(labels_batch.numpy())

            batch_df = pd.DataFrame({
                'embed1': [vec.tolist() for vec in embed1.cpu().numpy()],
                'embed2': [vec.tolist() for vec in embed2.cpu().numpy()],
                'output_all': [vec.tolist() for vec in outputs.cpu().numpy()],
                'function_pair': target_df['function_pair'].iloc[len(y_true) - len(labels_batch):len(y_true)],
                'file_pair': target_df['file_pair'].iloc[len(y_true) - len(labels_batch):len(y_true)],
                'label': labels_batch.numpy(),
                'prediction': batch_pred
            })

            chunk_results.append(batch_df)

            if len(y_true) >= chunk_size * (chunk_idx + 1):
                chunk_df = pd.concat(chunk_results, axis=0)
                chunk_df.to_csv(os.path.join(savePath, f'result_chunk_{chunk_idx}.csv'), index=False)
                chunk_results = []
                chunk_idx += 1
                print(f'Saved chunk {chunk_idx} to disk')
                torch.cuda.empty_cache()

            total_time += time.time() - start_time

    if chunk_results:
        chunk_df = pd.concat(chunk_results, axis=0)
        chunk_df.to_csv(os.path.join(savePath, f'result_chunk_{chunk_idx}.csv'), index=False)
        print(f'Saved final chunk {chunk_idx} to disk')
        torch.cuda.empty_cache()

    avg_time_per_sample = total_time / len(y_pred)

    result_df = pd.concat(
        [pd.read_csv(os.path.join(savePath, f)) for f in os.listdir(savePath) if f.startswith('result_chunk_')])

    print(f"\nEvaluation Metrics:")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"avg_time_per_sample:  {avg_time_per_sample:.10f}")

    valid_pairs = set()
    for _, row in result_df.iterrows():
        target, candidate = row['file_pair'].split('---')
        if target != candidate:
            valid_pairs.add(row['file_pair'])

    result_df[['func1', 'func2']] = result_df['function_pair'].str.split('---', expand=True)

    result_dict = {}
    for file_pair, group in tqdm(result_df.groupby('file_pair'), desc="Processing file pairs"):
        func_list = group[group['func1'] == group['func2']]['func1'].unique().tolist()
        if func_list:
            result_dict[file_pair] = func_list

    for pair_set in tqdm(valid_pairs, desc="Processing targets"):
        filtered_df = result_df[result_df['file_pair'] == pair_set]
        filtered_df = filtered_df[filtered_df['prediction'] == 1]
        if not filtered_df.empty:
            save_file_path = os.path.join(savePath, f'{pair_set}_dataset.csv')
            filtered_df.to_csv(save_file_path, index=False)

    output_file = os.path.join(savePath, "area_ground_truth.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f'Results saved to {savePath}')

def create_siamese_dataloader(target_X, candidate_X, y, batch_size=32, shuffle=False):
    """创建PyTorch数据加载器"""
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(target_X),
        torch.FloatTensor(candidate_X),
        torch.LongTensor(y)
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

def load_trained_model(model_path, input_dim):
    """加载PyTorch模型"""
    model = torch.load(model_path)
    model.eval()  # 设置为评估模式
    return model
