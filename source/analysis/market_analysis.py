import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..data.market import return_market_data
from datetime import datetime

# List of ECB crisis periods as tuples: (Label, Start Date, End Date)
# End dates set to None or current date represent ongoing or transitional periods
from datetime import datetime

# ECB crisis and shock periods as tuples: (Label, Start Date, End Date)
ecb_crises = [
    # Immediate response to the terrorist attacks in the US
    ("9/11 Liquidity Crisis", datetime(2001, 9, 11), datetime(2001, 10, 31)),
    
    # Financial turmoil leading into the major crash
    ("Financial Turmoil", datetime(2007, 8, 9), datetime(2008, 9, 14)),
    
    # Triggered by Lehman Brothers collapse
    ("Global Financial Crisis", datetime(2008, 9, 15), datetime(2010, 5, 1)),
    
    # European sovereign debt issues
    ("Eurozone Sovereign Debt Crisis", datetime(2010, 5, 2), datetime(2013, 8, 1)),
    
    # Pandemic emergency measures (PEPP)
    ("COVID-19 Pandemic Crisis", datetime(2020, 3, 1), datetime(2022, 2, 23)),
    
    # Post-invasion energy prices and rate hikes
    ("Energy Shock and Inflation Surge", datetime(2022, 2, 24), datetime.now())
]



data = return_market_data("Dataset_EA-MPD.xlsx")
data2 = return_market_data("shocks_ecb_mpd_me_d.csv")
data_big = pd.merge(data,data2.rename(columns={"STOXX50":"STOXX50_d"}),on="date").reset_index()
print(data_big)
ax = sns.lineplot(data=data_big, x="date", y=data_big["pc1"].abs().rolling(10).mean(), label="pc1")
green_to_yellow = ["#008000", "#55a630", "#80b918", "#aacc00", "#dddf00", "#ffff00"]
for (label, start,end), color in zip(ecb_crises,green_to_yellow):
    ax.axvspan(start, end, alpha=0.2, label=label,color=color)

sns.lineplot(data=data_big, x="date", y=data_big["pc1"].rolling(15).std(), label="pc1 std")
plt.legend()
plt.show()