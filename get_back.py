import pandas as pd

year = 2008
alg = "lr"
# alg = "xgb"
# 读取 CSV 文件
file_path = f"{year}_{alg}_pred.csv"
df = pd.read_csv(file_path, on_bad_lines='skip')

# 将 time 列转换为日期格式
df['time'] = pd.to_datetime(df['time'])

# 提取日期部分
df['date'] = df['time'].dt.date

# 定义一个函数来获取每一天 y_pred 最高的 10 行和最低的 10 行
def filter_top_bottom(df):
    top_10 = df.nlargest(10, 'y_pred')
    bottom_10 = df.nsmallest(10, 'y_pred')
    return pd.concat([top_10, bottom_10])

# 按日期分组并应用过滤函数
filtered_df = df.groupby('date').apply(filter_top_bottom).reset_index(drop=True)

# 删除临时的 date 列
# filtered_df.drop(columns=['date'], inplace=True)

# 显示结果
print(filtered_df)

filtered_df['result'] = filtered_df.apply(lambda row: ((row['p3'] - row['p2'])/row['p2']) if row['y_pred'] >= 0.5 else ((row['p2'] - row['p3'])/row['p2']), axis=1)

result_sum = filtered_df['result'].sum()
print("Sum of the 'result' column:", result_sum)

# 如果需要，将结果保存到新的 CSV 文件
filtered_df.to_csv(f"filtered_time_{year}_{alg}.csv", index=False)