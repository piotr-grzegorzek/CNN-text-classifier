from run import imdb, x_train

# Get the word index from the imdb dataset
word_index = imdb.get_word_index()

# Create a new dictionary where the keys are the indices and the values are the words
reverse_word_index = dict([(value, key)
                          for (key, value) in word_index.items()])


# Function to decode reviews
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# Display an example review
for i in range(5):
    print(decode_review(x_train[i]))
    print()
