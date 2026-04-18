import pandas as pd
from chunker import STATEMENTS_DIR
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_SELECTION = "finbert"

data = pd.read_csv(f"{STATEMENTS_DIR}/sentiment/{MODEL_SELECTION}_intro.csv", index_col="date").rename(
    columns={"score": "score_press"}
)
data_qa = pd.read_csv(f"{STATEMENTS_DIR}/sentiment/{MODEL_SELECTION}_qa.csv").pivot(index="date",columns="is_question")
data_qa.rename(columns={False:"Answer",True:"Question"},inplace=True)
data_qa.columns = [x[1] for x in list(data_qa.columns)]
# print(data_qa)
data[["score_answer","score_question"]] = data_qa[["Answer","Question"]]
data.reset_index(inplace=True)
data["date"] = pd.to_datetime(data["date"])
data["score"] = (data["score_press"] + 3 * data["score_answer"]) / 2
# mean_score = data["score"].mean()
# std_score = data["score"].std()
# data['z_score'] = (data['score'] - mean_score) / std_score
# print(data.describe())

# sns.lineplot(data=data, x="date", y="score")
# plt.axhline()
# plt.show()
melted_df = pd.melt(data.drop(columns=["score_press","score"]), id_vars=["date"], var_name="measure")
# print(mean)
print(data[data["score"] < -0.1])
sns.lineplot(data=melted_df, x="date", y="value",hue="measure")
plt.axhline()
plt.show()




# data.set_index("date").to_csv(f"{STATEMENTS_DIR}/sentiment/{MODEL_SELECTION}.csv")
