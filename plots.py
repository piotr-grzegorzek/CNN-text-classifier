from run import x_train, x_test
import matplotlib.pyplot as plt
import numpy as np


def create_histogram(data, color, label, subplot_position):
    plt.subplot(1, 3, subplot_position)
    plt.hist(
        data, bins=range(1, maxWords + 2, step), color=color, alpha=0.7, label=label
    )
    plt.title(f"{label} Histogram")
    plt.xlabel("Words count")
    plt.ylabel("Reviews count")
    plt.legend()


limit = None
step = 10

# Filter the reviews based on limit
train_lengths = [
    len(review) for review in x_train if limit is None or len(review) <= limit
]
test_lengths = [
    len(review) for review in x_test if limit is None or len(review) <= limit
]

maxWords = max(max(train_lengths), max(test_lengths))

# Create a figure with 3 subplots
plt.figure(figsize=(18, 6))

# First subplot: Training Reviews Histogram
create_histogram(train_lengths, "blue", "Training Reviews", 1)

# Second subplot: Testing Reviews Histogram
create_histogram(test_lengths, "green", "Testing Reviews", 2)

# Third subplot: Difference Histogram
# Calculate the difference in counts using histograms directly
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
