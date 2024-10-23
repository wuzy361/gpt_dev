from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from pdb import set_trace
import pandas as pd
import ast
import os
from sklearn.metrics import confusion_matrix
if __name__ == "__main__":

    # years = [2008,2009,2010,2011]
    years = [2012]
    X_list = []
    y_list = []
    X = None
    y = None
    for year in years:
        x_name = f"{year}_X.npy"
        y_name = f"{year}_y.npy"

        if os.path.exists(x_name) and os.path.exists(y_name):
            # 如果文件存在，从文件加载数据
            X_part = np.load(x_name)
            y_part = np.load(y_name)
            print(f"Loaded data from {x_name} and {y_name}")
            # set_trace()
        else:
            # 如果文件不存在，从CSV文件加载数据并保存
            train_path = f"train_{year}.csv"
            df = pd.read_csv(train_path, on_bad_lines='skip')
            df['emb'] = df['emb'].apply(lambda x: np.array(ast.literal_eval(x), dtype=float))
            X_part = np.array(df['emb'].tolist())
            y_part = np.array(df['tag'].tolist())

            np.save(x_name, X_part)
            np.save(y_name, y_part)
            print(f"Saved data to {x_name} and {y_name}")
        
        print("y_part_avg = " + str(np.mean(y_part)))
        X_list.append(X_part)
        y_list.append(y_part)

    X = np.concatenate(X_list,axis=0)
    y = np.concatenate(y_list,axis=0)
    print("y_avg = " + str(np.mean(y)))

    #为了让标签平均，删除一些sample
    class1_num = len(np.where(y == 1)[0])
    class0_num = len(np.where(y == 0)[0])
    need_delate_num = class1_num - class0_num
    
    if need_delate_num > 0:
        class_index = np.where(y == 1)[0]
    else:
        class_index = np.where(y == 0)[0]

    random_selection = np.random.choice(class_index, size=abs(need_delate_num), replace=False)

    X_new = np.delete(X, random_selection, axis=0)
    y_new = np.delete(y, random_selection, axis=0)

    print("y_new= " + str(np.mean(y_new)))
        
    set_trace()
    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

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
    model = SGDClassifier(loss='log_loss', max_iter=1, learning_rate='constant', eta0=0.01, random_state=42, verbose=1)

    # 训练模型，使用多个 epoch
    # n_epochs = 32 # 2012 使用32epoch 最好
    n_epochs= 30
    for epoch in range(n_epochs):
        model.partial_fit(X_train, y_train, classes=np.unique(y))
        print(f"Epoch {epoch + 1}/{n_epochs} completed.")

    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"Final Accuracy: {accuracy:.2f}")

    # 打印混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
