import pandas as pd
from .. import STATEMENTS_DIR

data = pd.read_csv(f"{STATEMENTS_DIR}/scraped_v2.csv")