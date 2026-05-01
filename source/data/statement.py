import pandas as pd
from .. import DATABASE_DIR, FILENAME

data = pd.read_csv(f"{DATABASE_DIR}/{FILENAME}.csv")