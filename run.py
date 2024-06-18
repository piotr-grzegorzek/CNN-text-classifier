import pandas as pd  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.layers import (  # type: ignore
    Dense,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dropout,
    SpatialDropout1D,
)
from keras.optimizers import Adam  # type: ignore
from keras._tf_keras.keras.preprocessing.text import Tokenizer  # type: ignore
from keras.preprocessing.sequence import pad_sequences  # type: ignore
from keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from keras.models import load_model  # type: ignore

max_features = 20000
max_len = 200
batch_size = 32
epochs = 10

# Load training data
train_data = pd.read_csv(
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    header=None,
)
train_data.columns = ["Category", "Title", "Description"]
train_data["Text"] = train_data["Title"] + " " + train_data["Description"]
x_train_data = train_data["Text"].astype(str)
y_train_data = train_data["Category"] - 1

# Load testing data
test_data = pd.read_csv(
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
    header=None,
)
test_data.columns = ["Category", "Title", "Description"]
test_data["Text"] = test_data["Title"] + " " + test_data["Description"]
x_test_data = test_data["Text"].astype(str)
y_test_data = test_data["Category"] - 1

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train_data)
x_train_seq = tokenizer.texts_to_sequences(x_train_data)
x_train_padded = pad_sequences(x_train_seq, maxlen=max_len)
x_test_seq = tokenizer.texts_to_sequences(x_test_data)
x_test_padded = pad_sequences(x_test_seq, maxlen=max_len)

# Create validation set from training set
x_train, x_val, y_train, y_val = train_test_split(
    x_train_padded, y_train_data, test_size=0.2
)

if __name__ == "__main__":

    model = Sequential()
    model.add(
        Embedding(
            max_features,
            output_dim=128,
            input_length=max_len,
        )
    )
    model.add(SpatialDropout1D(0.5))

    model.add(
        Conv1D(
            filters=256,
            kernel_size=5,
            activation="relu",
        )
    )
    model.add(Dropout(0.3))
    model.add(GlobalMaxPooling1D())
    model.add(
        Dense(
            units=32,
            activation="relu",
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(units=4, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(
            learning_rate=0.00049342,
        ),
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )

    # Modify the ModelCheckpoint callback to save only the best model
    model_checkpoint = ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[
            early_stopping,
            model_checkpoint,  # Use the modified ModelCheckpoint callback
        ],
    )

    best_model = load_model("best_model.keras")

    score = best_model.evaluate(x_test_padded, y_test_data)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Print the summary of the best model
    print(best_model.summary())

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()
