from matplotlib import pyplot as plt
from keras.datasets import imdb # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D # type: ignore
import utils

max_features = 20000
maxlen = 100
batch_size = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

if __name__ == "__main__":
    x_train, x_test = utils.pad_sequences(x_train, x_test, maxlen)
    model = Sequential()
    model.add(Embedding(max_features, output_dim=128, input_length=maxlen))
    model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=5,
        validation_data=(x_test, y_test),
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    plt.tight_layout()
    plt.show()

    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save("model.keras")
