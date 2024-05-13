import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

dataframe = pd.read_json('spotify_data.json')

print(dataframe.sample(n=10))
print()
print(dataframe.describe())