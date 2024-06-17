import pandas as pd  # type: ignore

data = pd.read_csv(
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    header=None,
)

data.columns = ["Category", "Title", "Description"]
category_map = {1: "World", 2: "Sport", 3: "Business", 4: "Tech"}

for i in range(500, 505):
    print(
        f"Category: {category_map[data['Category'][i]]}, Title: {data['Title'][i]}, Description: {data['Description'][i]}"
    )
    print()
