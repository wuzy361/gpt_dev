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
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def train_and_test(years, alg, save_res=False):
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
            print(f"Loaded data from {x_name} and {y_name}, size = {len(X_part)}")
            # set_trace()
        else:
            # 如果文件不存在，从CSV文件加载数据并保存
            train_path = f"train_{year}_new.csv"
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
    # need_delate_num = 0 
    
    if need_delate_num > 0:
        class_index = np.where(y == 1)[0]
    else:
        class_index = np.where(y == 0)[0]

    random_selection = np.random.choice(class_index, size=abs(need_delate_num), replace=False)

    X_new = np.delete(X, random_selection, axis=0)
    y_new = np.delete(y, random_selection, axis=0)

    print("len(y_new)= " + str(len(y_new)))
    print("y_new= " + str(np.mean(y_new)))
        
    # set_trace()
    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42, shuffle=False)
    # train_size = int(len(y_new) * 1)
    # val_size = int(len(y_new) * 0)
    # test_size = int(len(y_new) * 0)

    # X_train = X_new[:train_size]
    # y_train = y_new[:train_size]
    # train_size = X_train.shape[0]
    # # y_train = np.random.randint(0, 2, train_size)
    # # set_trace()
    # X_val = X_new[train_size:(train_size+val_size)]
    # y_val = y_new[train_size:(train_size +val_size)]
    # X_test= X_new[(train_size+val_size):]
    # y_test= y_new[(train_size+val_size):]

    

    # x_name = f"2023_X.npy"
    # y_name = f"2023_y.npy"
    # X_test = np.load(x_name)
    # y_test = np.load(y_name)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    # set_trace()
    # # 标准化数据
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    if alg == "xgb":
        # 构建DMatrix
        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        # val_dmatrix = xgb.DMatrix(X_val, label=y_val)
        test_dmatrix = xgb.DMatrix(X_test, label=y_test)

        # 设置参数
        params = {
            'objective': 'reg:squarederror',  # 预测目标是分类
            'max_depth': 5,
            'learning_rate': 0.05,
            # 'eval_metric': 'logloss',
            # 'n_estimators': 100,
            'alpha': 0.1,  # L1正则化项权重
            'lambda': 0.1  # L2正则化项权重
        }

        # 训练模型
        model = xgb.train(params, train_dmatrix, num_boost_round=200)

        # 测试模型
        y_pred_score = model.predict(test_dmatrix)
        y_pred = np.round(y_pred_score).astype(int) 

        # set_trace()

    if alg == "lr":
        # 创建 SGDClassifier 模型，使用逻辑回归（loss='log'）
        model = SGDClassifier(loss='log_loss', max_iter=1, learning_rate='constant', eta0=0.01, random_state=42, verbose=0)

        # 训练模型，使用多个 epoch
        # n_epochs = 32 # 2012 使用32epoch 最好
        n_epochs= 30
        for epoch in range(n_epochs):
            model.partial_fit(X_train, y_train, classes=np.unique(y))

            if epoch %10 == 0:
                print(f"Epoch {epoch + 1} completed.")
            #     # y_pred = model.predict(X_val)

            #     # 计算准确率
            #     accuracy = np.mean(y_pred == y_val)
            #     print(f"{epoch} Accuracy in : {accuracy:.4f}")
            #     conf_matrix = confusion_matrix(y_pred, y_val)
            #     print("Confusion Matrix:")
            #     print(conf_matrix)


        # 预测测试集
            # 预测测试集
        y_pred_score = model.predict_proba(X_test)[:, 1]  # 获取正类的概率

        # 打印预测的概率值
        print("Predicted probabilities for the test set:")
        print(y_pred_score)

        # 如果需要将概率转换为标签，可以使用阈值（例如0.5）
        y_pred = (y_pred_score >= 0.5).astype(int)
        print(y_pred)
        y_pred = model.predict(X_test)
        print(y_pred)
    
    if save_res:
        tag_df = pd.read_csv(f"time_{year}.csv", on_bad_lines='skip', skiprows=random_selection)
        # tag_df = pd.read_csv(f"time_2023.csv", on_bad_lines='skip')
        tag_df = tag_df[len(X_train):]
        # tag_df =
        y_pred_df = pd.DataFrame(y_pred_score, columns=['y_pred'])
        # set_trace()
        pre_df = pd.concat([tag_df.reset_index(drop=True), y_pred_df], axis=1)
        # pre_df.to_csv(f"{years[0]}_{alg}_pred.csv")
        pre_df.to_csv(f"{years[0]}_{alg}_pred.csv")
        # time_df.loc[x] = [newsid, time, symbol, t2, p2, t3, p3]
    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"{year} {alg} Final Accuracy: {accuracy:.4f}")

    # 打印混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"{year} {alg}Confusion Matrix:")
    print(conf_matrix)

if __name__ == "__main__":

    # years = [2020, 2021, 2022, 2023]
    train_and_test([2008], "lr", True)
    # train_and_test([2020], "lr")
    # train_and_test([2020,2021,2022], "lr", True)
    # train_and_test([2023], "lr")
