import pandas as pd
from openai import OpenAI
from pdb import set_trace
import numpy as np
import ast



if __name__ == "__main__":
#    years = [2009,2010,2011]
#    years = [2021,2022,2023]
   years = [2008]

   for year in years:
       data_path = "data_" + str(year) + ".csv" 
       df = pd.read_csv(data_path,on_bad_lines='skip')
       data_num = len(df)
       time_df = pd.DataFrame(columns=['newsid', 'time', "symbol" , "t2", "p2", "t3", "p3"])
       set_trace()
       print(data_num)

       for x in range(data_num):
        #    set_trace()
           newsid =  df.iloc[x,0]
           time =  df.iloc[x,1]
           symbol =  df.iloc[x,3]
           t2 =  df.iloc[x,7]
           p2 =  df.iloc[x,8]
           t3 =  df.iloc[x,9]
           p3 =  df.iloc[x,10]
           time_df.loc[x] = [newsid, time, symbol, t2, p2, t3, p3]
           if x % 10000 == 0:
               print(x)
           
       time_df.to_csv(f'time_{year}.csv', index=False,  encoding='utf-8')
       print(f"save {year}  data time")
      
#    set_trace()
