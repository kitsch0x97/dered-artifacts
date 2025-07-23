import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import pdb

import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import time
import numpy as np
import os

global_ratio = 10
global_threshold = 0.5

def custom_lower_fpr_train(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")  # 真实值和预测值的差距
    # 计算梯度
    grad = np.where(residual > 0, -1 * global_ratio * residual, -1 * residual)  # 对误报加大惩罚（假阳性）
    # 计算Hessian（第二阶导数）
    hess = np.where(residual > 0, global_ratio, 1)  # 对误报加大惩罚的Hessian
    return grad, hess

# 自定义评估指标：用于监控训练过程中的模型性能
def custom_eval_metric(y_true, y_pred):
    # 这里可以定义任何你需要的评估指标
    recall = recall_score(y_true, (y_pred > global_threshold))  # 计算召回率
    # print(f"recall:{recall}")
    return 'custom_recall', recall, True  # 返回自定义召回率

def train_lightgbm_v2(X_train, X_test, y_train, y_test, model_filename):

    model_name = 'lightgbm_model.txt'

    save_path = os.path.join(model_filename, model_name)

    # 创建LightGBM的LGBMRegressor模型
    gbm = lgb.LGBMClassifier(  # 对于二分类任务，使用LGBMClassifier
        objective=custom_lower_fpr_train,  # 使用自定义损失函数
        metric='binary_error',  # 用于二分类的评估指标
        boosting_type='gbdt',  # 使用梯度提升树（GBDT）
        num_leaves=31,  # 树的最大叶子数
        learning_rate=0.05,  # 学习率
        feature_fraction=0.9,  # 每次迭代使用的特征比例
        is_unbalance=True,  # 使用不平衡的类别样本
        verbose=-1,  # 禁用输出
        n_estimators=500
    )

    # 训练模型，传入自定义评估指标
    gbm.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],  # 传入验证集
        eval_metric=custom_eval_metric  # 传入自定义评估指标
    )

    gbm.booster_.save_model(save_path)

    # 预测测试集
    start_time = time.time()
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    end_time = time.time()
    execution_time = end_time - start_time
    y_pred_binary = [1 if i > global_threshold else 0 for i in y_pred]  # 转换为二分类标签

    # 计算准确率和召回率
    accuracy = accuracy_score(y_test, y_pred_binary)

    return gbm, accuracy, execution_time  # 返回训练好的模型、准确率、召回率和执行时间

def train_lightgbm(X_train, X_test, y_train, y_test, model_filename):

    # 创建LightGBM的数据格式
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 定义LightGBM参数
    params = {
        'objective': 'binary',           # 二分类任务
        'metric': 'binary_error',        # 用于二分类的评估指标
        'boosting_type': 'gbdt',         # 使用梯度提升树（GBDT）
        'num_leaves': 31,                # 树的最大叶子数
        'learning_rate': 0.05,           # 学习率
        'feature_fraction': 0.9,         # 每次迭代使用的特征比例
        'verbose': -1                    # 设置为 -1 禁用所有输出
    }

    # 训练模型
    model = lgb.train(params, train_data, num_boost_round=300, valid_sets=[test_data])
    model.save_model(model_filename)

    # 预测测试集
    # 评估模型
    start_time = time.time()
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    end_time = time.time()
    execution_time = end_time - start_time
    y_pred_binary = [1 if i > 0.5 else 0 for i in y_pred]  # 转换为二分类标签
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred_binary)

    return model, accuracy, execution_time  # 返回训练好的模型