import keras
import numpy as np
from run import x_train, x_test, y_test, maxlen
import utils

x_train, x_test = utils.pad_sequences(x_train, x_test, maxlen)

# Load the model
model = keras.models.load_model("model.keras")

# Make predictions
predictions = model.predict(x_test)

# Find the indices of the wrong predictions
wrong_predictions = np.where(predictions != y_test)[0]

# Print out the details of the wrong predictions
for i in wrong_predictions:
    index = wrong_predictions[i]
    print(f"Recenzja numer {index}")
    print(f"Rzeczywista etykieta: {y_test[index]}")
    print(f"Przewidziana etykieta: {predictions[index][0]}")
    print("\n")
