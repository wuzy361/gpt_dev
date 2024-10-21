from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from pdb import set_trace
import pandas as pd
import ast
import os
if __name__ == "__main__":

    years = [2008]
    for year in years:
        x_name = f"{year}_X.npy"
        y_name = f"{year}_y.npy"
        X = None
        y = None

        if os.path.exists(x_name) and os.path.exists(y_name):
            # 如果文件存在，从文件加载数据
            X = np.load(x_name)
            y = np.load(y_name)
            print(f"Loaded data from {x_name} and {y_name}")
        else:
            # 如果文件不存在，从CSV文件加载数据并保存
            train_path = f"train_{year}.csv"
            df = pd.read_csv(train_path, on_bad_lines='skip')
            df['emb'] = df['emb'].apply(lambda x: np.array(ast.literal_eval(x), dtype=float))
            X = np.array(df['emb'].tolist())
            y = np.array(df['tag'].tolist())

            np.save(x_name, X)
            np.save(y_name, y)
            print(f"Saved data to {x_name} and {y_name}")

        # 分割数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        
        # set_trace()
        # # 标准化数据
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        # 创建 SGDClassifier 模型，使用逻辑回归（loss='log'）
        model = SGDClassifier(loss='hinge', max_iter=100, learning_rate='constant', eta0=0.01, random_state=42, verbose=1)

        # 训练模型，使用多个 epoch
        n_epochs = 100
        for epoch in range(n_epochs):
            model.partial_fit(X_train, y_train, classes=np.unique(y))
            print(f"Epoch {epoch + 1}/{n_epochs} completed.")

        # 预测测试集
        y_pred = model.predict(X_test)

        # 计算准确率
        accuracy = np.mean(y_pred == y_test)
        print(f"Final Accuracy: {accuracy:.2f}")
