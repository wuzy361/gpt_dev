API_KEY = "sk-XbtNzC4S3dzClQWP06D1B18d72Cf45B2A77cD7A129BaF43d"
BASE_URL = "https://api.kwwai.top/v1"
TOKEN_NUM = 512 
EMB_SIZE = 512

# import pandas as pd
# from openai import OpenAI
# from pdb import set_trace
# import numpy as np
# client = OpenAI(
#      base_url= BASE_URL,
#      api_key= API_KEY
#  )

# def get_embedding(text, model="text-embedding-3-small"):
#    text = text.replace("\n", " ")
#    client_data = client.embeddings.create(input = [text], model=model)
#    return client_data.data[0].embedding[:512]

# if __name__ == "__main__":
#    data_path = "data_2008.csv"
#    df = pd.read_csv(data_path,on_bad_lines='skip')
# #    df['emb'] = [0.0]*512
#    df['emb'] = [np.zeros(512).tolist() for _ in range(len(df))]
# #    set_trace()
#    for x in range(len(df)):
#       print(x)
#       embeding = get_embedding(df.iloc[x,-2][:TOKEN_NUM])
#     #   set_trace()
#       df.at[x,'emb'] = embeding

#       if x % 500 == 0:
#         print(x)
#         df.to_csv(f'data_2008_emb_{x}.csv', index=False,  encoding='utf-8')
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
    embedding = get_embedding(row['content'][:TOKEN_NUM])
    return index, embedding

def process_main(data_path):
    # df = pd.read_csv(data_path, skiprows=range(1, 40000),  on_bad_lines='skip')
    # df = pd.read_csv(data_path,  on_bad_lines='skip',skiprows=range(1, 100))
    df = pd.read_csv(data_path,  on_bad_lines='skip')

    # Ensure the 'emb' column is initialized
    df['emb'] = [np.zeros(512).tolist() for _ in range(len(df))]

    # Constants
   #  TOKEN_NUM = 100  # Adjust as needed
    BATCH_SIZE = 5000
    THREADS = 30  # Adjust based on your system capabilities

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(process_row, i, row): i for i, row in df.iterrows()}

        for future in as_completed(futures):
            index, embedding = future.result()
            df.at[index, 'emb'] = embedding
            print(index)

            if index % BATCH_SIZE == 0 and index != 0:
                print(index)
                df.to_csv(data_path.split(".")[0] + "_" + str(index) + "_with_emb_last.csv", index=False, encoding='utf-8')

    df.to_csv(data_path.split(".")[0] + '_with_emb_last.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    # data_path = ["data_2017.csv","data_2018.csv","data_2019.csv"]
    data_path = ["data_2019.csv"]
    for x in data_path:
        process_main(x)
        time.sleep(20)
