import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load training data
train_data = pd.read_csv(
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    header=None,
)
train_data.columns = ["Category", "Title", "Description"]
train_data["Text"] = train_data["Title"] + " " + train_data["Description"]

# Load testing data
test_data = pd.read_csv(
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
    header=None,
)
test_data.columns = ["Category", "Title", "Description"]
test_data["Text"] = test_data["Title"] + " " + test_data["Description"]

# Calculate lengths of articles
limit = None  # Set limit to desired value or None for no limit
train_lengths = [
    len(text.split())
    for text in train_data["Text"]
    if limit is None or len(text.split()) <= limit
]
test_lengths = [
    len(text.split())
    for text in test_data["Text"]
    if limit is None or len(text.split()) <= limit
]

maxWords = max(max(train_lengths), max(test_lengths))
step = 1

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.hist(
    train_lengths,
    bins=range(1, maxWords + 2, step),
    color="blue",
    alpha=0.7,
    label="Training Articles",
)
plt.title("Training Articles Histogram")
plt.xlabel("Words count")
plt.ylabel("Articles count")
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(
    test_lengths,
    bins=range(1, maxWords + 2, step),
    color="green",
    alpha=0.7,
    label="Testing Articles",
)
plt.title("Testing Articles Histogram")
plt.xlabel("Words count")
plt.ylabel("Articles count")
plt.legend()

train_hist, bin_edges = np.histogram(train_lengths, bins=range(1, maxWords + 2, step))
test_hist, _ = np.histogram(test_lengths, bins=range(1, maxWords + 2, step))
diff_hist = train_hist - test_hist

plt.subplot(1, 3, 3)
plt.bar(
    bin_edges[:-1], diff_hist, width=step, color="purple", alpha=0.7, label="Difference"
)
plt.title("Difference Histogram")
plt.xlabel("Words count")
plt.ylabel("Articles count")
plt.legend()
plt.show()
