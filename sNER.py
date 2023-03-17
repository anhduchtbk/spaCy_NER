import pandas as pd

raw_df = pd.read_csv('./mtsamples.csv', index_col=0)
# print(raw_df.to_string())
raw_df.head()