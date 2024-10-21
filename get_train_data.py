import pandas as pd
from openai import OpenAI
from pdb import set_trace
import numpy as np



if __name__ == "__main__":
   data_list = [2008,2009,2010,2011]
   data_path = ["data_" + str(x) + "_with_emb.csv" for x in data_list]
   df = pd.read_csv(data_path[0],on_bad_lines='skip')
   data_num = len(df)
   train_df = pd.DataFrame(columns=['emb', 'tag'])
   set_trace()
