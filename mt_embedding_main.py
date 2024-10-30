API_KEY = "sk-oR2HFE7ZngYGjOvX748749B0B3D14fD6Bb119083735bC0F1"
BASE_URL = "https://api.wlai.vip/v1"
TOKEN_NUM = 512 
EMB_SIZE = 512

import pandas as pd
from openai import OpenAI
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    client_data = client.embeddings.create(input=[text], model=model, timeout=60)
    return client_data.data[0].embedding[:512]

def process_row(index, row ):
    print("send " + str(index))
    embedding = get_embedding(row['content'][:TOKEN_NUM])
    print("recive " + str(index))
    return index, embedding



def process_main(data_path):
    jump_lines = 200000
    # df = pd.read_csv(data_path, skiprows=range(1, jump_lines),  on_bad_lines='skip')
    # df = pd.read_csv(data_path,  on_bad_lines='skip',skiprows=range(1, 100))
    # 2021,2022 需要增加140000后的数据
    df = pd.read_csv(data_path,  on_bad_lines='skip')

    # Ensure the 'emb' column is initialized
    df['emb'] = [np.zeros(512).tolist() for _ in range(len(df))]

    # Constants
   #  TOKEN_NUM = 100  # Adjust as needed
    BATCH_SIZE = 20000
    THREADS = 50  # Adjust based on your system capabilities
    print("start")
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(process_row, i, row): i for i, row in df.iterrows()}

        for future in as_completed(futures):
            index, embedding = future.result()
            df.at[index, 'emb'] = embedding
            print(index)

            if index % BATCH_SIZE == 0 and index != 0:
                print(index)
                df.to_csv(data_path.split(".")[0] + "_" + str(index + jump_lines) + "_with_emb_last.csv", index=False, encoding='utf-8')

    df.to_csv(data_path.split(".")[0] + '_with_emb_final.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    # data_path = ["data_2017.csv","data_2018.csv","data_2019.csv"]
    data_path = ["data_2019.csv"]
    for x in data_path:
        process_main(x)
        time.sleep(20)

    # test = get_embedding("还好呀")
    # print(test)
