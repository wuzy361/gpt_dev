API_KEY = "sk-HHCb4mYKPg0GUcNN4fDc3c34B90648B89d874fFd138f33C3"
BASE_URL = "https://api.kwwai.top/v1"
TOKEN_NUM = 512
EMB_SIZE = 512

import pandas as pd
from openai import OpenAI
from pdb import set_trace
import numpy as np
client = OpenAI(
     base_url= BASE_URL,
     api_key= API_KEY
 )

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   client_data = client.embeddings.create(input = [text], model=model)
   return client_data.data[0].embedding[:512]

if __name__ == "__main__":
   data_path = "data_2008.csv"
   df = pd.read_csv(data_path,on_bad_lines='skip')
#    df['emb'] = [0.0]*512
   df['emb'] = [np.zeros(512).tolist() for _ in range(len(df))]
#    set_trace()
   for x in range(len(df)):
      print(x)
      embeding = get_embedding(df.iloc[x,-2][:TOKEN_NUM])
    #   set_trace()
      df.at[x,'emb'] = embeding

      if x % 500 == 0:
        print(x)
        df.to_csv(f'data_2008_emb_{x}.csv', index=False,  encoding='utf-8')
