import pandas as pd
from openai import OpenAI
from pdb import set_trace
import numpy as np
import ast



if __name__ == "__main__":
#    years = [2009,2010,2011]
   years = [2012]

   for year in years:
       data_path = "data_" + str(year) + "_with_emb.csv" 
       df = pd.read_csv(data_path,on_bad_lines='skip')
       data_num = len(df)
       train_df = pd.DataFrame(columns=['emb', 'tag'])

       for x in range(data_num):
           emb_str = df.iloc[x, -1]
        #    emb = ast.literal_eval(emb_str)
        #    emb = np.array(emb, dtype=np.float32)
           p1 =  df.iloc[x,6]
           p2 =  df.iloc[x,8]
           delta = p2 - p1
           tag = 1 if delta > 0 else 0
           train_df.loc[x] = [emb_str, tag]
           if x % 1000 == 0:
               print(x)
           
       train_df.to_csv(f'train_{year}.csv', index=False,  encoding='utf-8')
       print(f"save {year} train data")
      
#    set_trace()
