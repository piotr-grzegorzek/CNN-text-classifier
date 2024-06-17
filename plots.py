import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("path_to_AG_News_data.csv")
data["text"] = data["Title"] + " " + data["Description"]
x_data = data["text"]

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_data)
x_data_seq = tokenizer.texts_to_sequences(x_data)
x_data_padded = pad_sequences(x_data_seq, maxlen=max_len)

x_train, x_test, y_train, y_test = train_test_split(
    x_data_padded, data["Category"], test_size=0.2, random_state=42
)

train_lengths = [len(seq) for seq in x_train]
test_lengths = [len(seq) for seq in x_test]

maxWords = max(max(train_lengths), max(test_lengths))
step = 10

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.hist(
    train_lengths,
    bins=range(1, maxWords + 2, step),
    color="blue",
    alpha=0.7,
    label="Training Reviews",
)
plt.title("Training Reviews Histogram")
plt.xlabel("Words count")
plt.ylabel("Reviews count")
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(
    test_lengths,
    bins=range(1, maxWords + 2, step),
    color="green",
    alpha=0.7,
    label="Testing Reviews",
)
plt.title("Testing Reviews Histogram")
plt.xlabel("Words count")
plt.ylabel("Reviews count")
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
plt.ylabel("Difference in Reviews count")
plt.legend()

plt.tight_layout()
plt.show()
