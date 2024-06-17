import pandas as pd
import numpy as np
from keras.models import load_model  # type: ignore
from keras.preprocessing.text import Tokenizer  # type: ignore
from keras.preprocessing.sequence import pad_sequences  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

max_features = 20000
max_len = 100

data = pd.read_csv(
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    header=None,
)
data.columns = ["Category", "Title", "Description"]
data["Text"] = data["Title"] + " " + data["Description"]
x_data = data["Text"]
y_data = data["Category"] - 1  # Adjust categories to be 0-based

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_data)
x_data_seq = tokenizer.texts_to_sequences(x_data)
x_data_padded = pad_sequences(x_data_seq, maxlen=max_len)

x_train, x_test, y_train, y_test = train_test_split(
    x_data_padded, y_data, test_size=0.2, random_state=42
)

model = load_model("model.keras")

score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)

wrong_predictions = np.where(predictions != y_test)[0]

print(
    f"The percentage of wrong predictions is {(len(wrong_predictions) / len(predictions)) * 100}%"
)

for i in wrong_predictions[:5]:
    print(f"Review number {i}")
    print(f"Actual label: {y_test.iloc[i]}")
    print(f"Review text: {data['Text'].iloc[i]}")
    print(f"Predicted label: {predictions[i]}")
    print("\n")
