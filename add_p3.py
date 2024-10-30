import pandas as pd
from openai import OpenAI
from pdb import set_trace
import numpy as np
import ast



if __name__ == "__main__":
#    years = [2009,2010,2011]
   years = [2008]

   for year in years:
       data_path = "data_" + str(year) + "_with_emb.csv" 
       df = pd.read_csv(data_path,on_bad_lines='skip')
       df_with_p3 = pd.read_csv(f"data_{year}.csv",on_bad_lines='skip') 

       data_num = len(df)
    #    assert(len(df) == len(df_with_p3))
       train_df = pd.DataFrame(columns=['emb', 'tag'])
       df = df.merge(df_with_p3[['newsid','t3','p3']],on='newsid',how='inner')
    #    set_trace()

       for x in range(data_num):
           emb_str = df.iloc[x, -3]
        #    emb = ast.literal_eval(emb_str)
        #    emb = np.array(emb, dtype=np.float32)
           p1 =  df.iloc[x,6]
           p2 =  df.iloc[x,8]
           p3 =  df.iloc[x,-1]
           delta = p3 - p2
           tag = 1 if delta > 0 else 0
           train_df.loc[x] = [emb_str, tag]
           if x % 1000 == 0:
               print(x)
           
       train_df.to_csv(f'train_{year}_new.csv', index=False,  encoding='utf-8')
       print(f"save {year} train data")
      
#    set_trace()
