import keras
import numpy as np
from run import x_train, x_test, y_test, max_len, imdb
import utils

x_train, x_test = utils.pad_sequences(x_train, x_test, max_len)

model = keras.models.load_model("model.keras")
print(model.summary())

score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

predictions = model.predict(x_test)

predictions = [type(y)(p) for p, y in zip(predictions.flatten(), y_test)]

assert all(isinstance(p, type(y)) for p, y in zip(predictions, y_test))

wrong_predictions = np.where(predictions != y_test)[0]

word_index = imdb.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


percentage_wrong = (len(wrong_predictions) / len(predictions)) * 100
print(f"The percentage of wrong predictions is {percentage_wrong}%")

for i in wrong_predictions[10:15]:
    print(f"Review number {i}")
    print(f"Actual label: {y_test[i]}")
    print(f"Review text: {decode_review(x_test[i])}")
    print(f"Predicted label: {predictions[i]}")
    print("\n")
