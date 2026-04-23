import pandas as pd

from market import return_market_data

data = return_market_data()
date = pd.to_datetime('2013-07-04')
print(data.query("date == @date"))