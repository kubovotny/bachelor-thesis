import pandas as pd
from chunker import STATEMENTS_DIR
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(f"{STATEMENTS_DIR}/finbert_sentiment.csv", index_col="date").rename(
    columns={"score": "score_press"}
)
data_qa = pd.read_csv(f"{STATEMENTS_DIR}/finbert_sentiment_qa.csv").pivot(index="date",columns="is_question")
data_qa.rename(columns={False:"Answer",True:"Question"},inplace=True)
data_qa.columns = [x[1] for x in list(data_qa.columns)]
print(data_qa)
data[["score_answer","score_question"]] = data_qa[["Answer","Question"]]
data.reset_index(inplace=True)
data["date"] = pd.to_datetime(data["date"])
data["score"] = data["score_press"] + data["score_answer"]
mean = data["score"].mean()
data = pd.melt(data, id_vars=["date"], var_name="measure")
# print(mean)
# print(data[data["score"] < -0.1])
sns.lineplot(data=data, x="date", y="value",hue="measure")
plt.axhline()
plt.show()
