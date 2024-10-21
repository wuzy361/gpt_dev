import os
from pdb import set_trace
import pandas as pd
from datetime import datetime, timedelta


if __name__ == "__main__":
    # text = get_embedding("I am a fish, I am a baby fish")
    # print(text)
    base_path = "/Users/jerry/code/gpt_dev/"

    df1 =  pd.read_csv(base_path + "data_2009_emb_mt_20000.csv",on_bad_lines='skip',nrows=20000)
    df2 =  pd.read_csv(base_path + "data_2009_with_emb_mid.csv",on_bad_lines='skip')
    df3 =  pd.read_csv(base_path + "data_2009_with_emb_last.csv",on_bad_lines='skip')
    df_whole = pd.concat([df1, df2, df3], ignore_index=True)
    df_whole.to_csv(f'data_2009_with_emb.csv', index=False,  encoding='utf-8')
