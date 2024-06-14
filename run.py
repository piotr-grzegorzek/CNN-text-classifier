from keras.datasets import imdb  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout  # type: ignore
from keras.preprocessing.sequence import pad_sequences  # type: ignore
from keras_tuner import HyperModel, RandomSearch
from keras.optimizers import Adam, RMSprop, SGD  # type: ignore

max_features = 20000
max_len = 2000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

if __name__ == "__main__":

    class IMDBHyperModel(HyperModel):
        def build(self, hp):
            model = Sequential()

            # Embedding layer
            model.add(
                Embedding(
                    max_features,
                    output_dim=hp.Int(
                        "embedding_output_dim", min_value=32, max_value=128, step=32
                    ),
                    input_length=max_len,
                )
            )

            # Conv1D layers with dropout
            for i in range(hp.Int("num_conv_layers", 1, 3)):
                model.add(
                    Conv1D(
                        filters=hp.Int(
                            f"filters_{i}", min_value=32, max_value=128, step=32
                        ),
                        kernel_size=hp.Choice(f"kernel_size_{i}", values=[3, 5, 7]),
                        activation="relu",
                    )
                )
                model.add(
                    Dropout(
                        rate=hp.Float(
                            f"dropout_{i}", min_value=0.1, max_value=0.5, step=0.1
                        )
                    )
                )

            model.add(GlobalMaxPooling1D())

            # Dense layer with dropout
            model.add(
                Dense(
                    units=hp.Int("hidden_units", min_value=32, max_value=128, step=32),
                    activation="relu",
                )
            )
            model.add(
                Dropout(
                    rate=hp.Float(
                        "dropout_dense", min_value=0.1, max_value=0.5, step=0.1
                    )
                )
            )
            model.add(Dense(units=1, activation="sigmoid"))

            # Tuning optimizer
            optimizer = hp.Choice("optimizer", ["adam", "rmsprop", "sgd"])
            if optimizer == "adam":
                opt = Adam()
            elif optimizer == "rmsprop":
                opt = RMSprop()
            else:
                opt = SGD()

            model.compile(
                loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]
            )

            return model

    def pad_data(maxlen):
        return pad_sequences(x_train, maxlen=maxlen), pad_sequences(
            x_test, maxlen=maxlen
        )

    tuner = RandomSearch(
        IMDBHyperModel(),
        objective="val_loss",
        max_trials=20,
        executions_per_trial=1,
        directory="my_dir",
        project_name="imdb_reviews_extended",
    )

    x_train_padded, x_test_padded = pad_data(max_len)

    tuner.search(
        x_train_padded,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_test_padded, y_test),
        callbacks=[],
    )

    best_model = tuner.get_best_models(num_models=1)[0]

    score = best_model.evaluate(x_test_padded, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    best_model.save("model.keras")
