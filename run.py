import matplotlib.pyplot as plt
from keras.datasets import imdb # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
from keras_tuner import HyperModel, RandomSearch

max_features = 20000
maxlen = 800
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


class IMDBHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(
            Embedding(
                max_features,
                output_dim=hp.Int(
                    "embedding_output_dim", min_value=32, max_value=128, step=32
                ),
                input_length=maxlen,
            )
        )
        model.add(
            Conv1D(
                filters=hp.Int("filters", min_value=32, max_value=128, step=32),
                kernel_size=hp.Choice("kernel_size", values=[3, 5, 7]),
                activation="relu",
            )
        )
        model.add(GlobalMaxPooling1D())
        model.add(Dense(units=1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model


tuner = RandomSearch(
    IMDBHyperModel(),
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    directory="my_dir",
    project_name="imdb_reviews",
)

tuner.search_space_summary()

tuner.search(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

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

score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save("model.keras")
