import os
from pdb import set_trace
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor


# 处理新闻证券关联表
def process_map_file(news_map_path):
    # 读取CSV文件
    df = pd.read_csv(news_map_path,on_bad_lines='skip')

    # 过滤掉不是 P50100 的行
    filtered_df = df[df['SecurityTypeID'] == 'P50100']

    # 找出重复的 NewsID
    duplicate_ids = filtered_df[filtered_df.duplicated('NewsID', keep=False)]['NewsID'].unique()

    # 删除这些重复的 NewsID
    result_df = filtered_df[~filtered_df['NewsID'].isin(duplicate_ids)]

    return result_df


def process_news_file(news_path):
    # 读取CSV文件
    df = pd.read_csv(news_path,on_bad_lines='skip')
    return df

def symbol_complete(symbol):
    while len(symbol) < 6:
        symbol = '0' + symbol
    if symbol[0] == '0' or symbol[0] == '3':
        symbol = "sz."+ symbol
    elif symbol[0] == '6' or symbol[0] == '9':
        symbol = "sh."+ symbol
    return symbol
        


def get_stock_price_at_time(price_path, symbol, time_str):
 
    time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    # 开盘时间
    comparison_time = datetime.strptime('09:30:00', '%H:%M:%S').time()
    # 获取 time_obj 的时间部分
    time_obj_time = time_obj.time()
    offset = 0
    # 比较时间
    if time_obj_time < comparison_time:
        offset =  1
    delta = timedelta(days=offset)
    new_time_obj = time_obj - delta
    new_time_obj = new_time_obj.strftime('%Y-%m-%d')

    
    # 遍历目录中的所有文件
    full_symbol =  symbol_complete(str(symbol))
    file_path = os.path.join(price_path, f"{full_symbol}.csv")

    # 检查文件是否存在，然后读取 CSV 文件
    result = 0
    if os.path.exists(file_path):
        # print(file_path)
        df = pd.read_csv(file_path,encoding="GB2312",on_bad_lines='skip')

        index = df[df["日期"] <= new_time_obj].tail(1).index
        if len(index) <1:
            print("缺少股票数据")
            return (None,None,None,None)
        # set_trace()
        result = df.iloc[index[0]:index[0] + 2]
    else:
        print(f"文件 {file_path} 不存在")
        return (None,None,None,None)

    if len(result) == 2:
        t1 = result.iloc[0,0]
        t2 = result.iloc[1,0]
        p1 = result.iloc[0,2]
        p2 = result.iloc[1,2]
        return (t1,p1,t2,p2)
    else:
        return (None,None,None,None)


# if __name__ == "__main__":
#     # text = get_embedding("I am a fish, I am a baby fish")
#     # print(text)
#     years = [2012,2013,2014,2015]
#     for year in years:
#         base_path = "/Users/jerry/code/股票预测/"
#         df1 = process_map_file(base_path + "CSMAR新闻数据/新闻证券关联表2012-2015.csv")
#         df2 = process_news_file(base_path + f"CSMAR新闻数据/{year}.csv")
#         # set_trace()

#         common_news = pd.merge(df1, df2, on='NewsID')
#         common_news = common_news.dropna(subset=["NewsContent"])
#         news_num = len(common_news)
        
#         # set_trace()
#         data_df = pd.DataFrame(columns=['newsid',  'time', 'title', 'symbol', 'shortname', 't1','p1','t2', 'p2', 'tag', 'length', 'content' ])
#         # set_trace()
#         for x in range(news_num):
#             print(x)
#             symbol = common_news.iloc[x,3]
#             time = common_news.iloc[x,7]
#             price_path = base_path + "不复权"
#             t1,p1,t2,p2 = get_stock_price_at_time(price_path, symbol, time)
#             if t1 == None:
#                 print("found None")
#                 continue
#             newsid = common_news.iloc[x,0]
#             time = common_news.iloc[x,10]
#             title = common_news.iloc[x,2]
#             symbol = symbol_complete(str(common_news.iloc[x,3]))
#             shortname = common_news.iloc[x,4]
#             content = common_news.iloc[x,14]
#             if type(content) is not str:
#                 print("no content")
#                 continue
#             length = len(content)
#             tag = None
#             if p2 - p1 >= 0:
#                 tag = 1
#             else:
#                 tag = 0
#             # set_trace()
#             data_df.loc[x] = [newsid, time, title, symbol, shortname, t1, p1, t2, p2,  tag, length, content]
        
#         data_df.to_csv(f'data_{year}.csv', index=False,  encoding='utf-8')




def process_year(year, base_path):
    df1 = process_map_file(base_path + "CSMAR新闻数据/新闻证券关联表2020-2023.csv")
    df2 = process_news_file(base_path + f"CSMAR新闻数据/{year}.csv")

    common_news = pd.merge(df1, df2, on='NewsID')
    common_news = common_news.dropna(subset=["NewsContent"])
    news_num = len(common_news)

    data_df = pd.DataFrame(columns=['newsid', 'time', 'title', 'symbol', 'shortname', 't1', 'p1', 't2', 'p2', 'tag', 'length', 'content'])

    for x in range(news_num):
        if x % 1000 == 0:
            print(f"{year} {x}") 
        symbol = common_news.iloc[x, 3]
        time = common_news.iloc[x, 7]
        price_path = base_path + "不复权"
        t1, p1, t2, p2 = get_stock_price_at_time(price_path, symbol, time)
        if t1 is None:
            continue
        newsid = common_news.iloc[x, 0]
        time = common_news.iloc[x, 10]
        title = common_news.iloc[x, 2]
        symbol = symbol_complete(str(common_news.iloc[x, 3]))
        shortname = common_news.iloc[x, 4]
        content = common_news.iloc[x, 14]
        if type(content) is not str:
            continue
        length = len(content)
        tag = 1 if p2 - p1 >= 0 else 0
        data_df.loc[x] = [newsid, time, title, symbol, shortname, t1, p1, t2, p2, tag, length, content]

    output_path = f'data_{year}.csv'
    data_df.to_csv(output_path, index=False, encoding='utf-8')
    return output_path

if __name__ == "__main__":
    years = [2020,2021,2022,2023]
    base_path = "/Users/jerry/code/股票预测/"
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_year, year, base_path) for year in years]
        for future in futures:
            try:
                result = future.result()
                print(f"Processed file: {result}")
            except Exception as e:
                print(f"Error processing file: {e}")


    
    # set_trace()