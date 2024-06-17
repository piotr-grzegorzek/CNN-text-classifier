from keras.datasets import imdb  # type: ignore
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
from keras.preprocessing.sequence import pad_sequences  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
from keras_tuner import HyperModel
from keras_tuner.tuners import Hyperband

max_features = 20000
max_len = 2000
batch_size = 32
epochs = 10

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

if __name__ == "__main__":
    x_train_padded = pad_sequences(x_train, maxlen=max_len)
    x_test_padded = pad_sequences(x_test, maxlen=max_len)

    class IMDBHyperModel(HyperModel):
        def build(self, hp):
            model = Sequential()
            model.add(
                Embedding(
                    max_features,
                    hp.Int(
                        "embedding_output_dim", min_value=32, max_value=256, step=32
                    ),
                    input_length=max_len,
                )
            )
            model.add(
                SpatialDropout1D(
                    hp.Float("spatial_dropout", min_value=0.1, max_value=0.5, step=0.1)
                )
            )
            for i in range(hp.Int("num_conv_layers", 1, 3)):
                model.add(
                    Conv1D(
                        filters=hp.Int(
                            f"filters_{i}", min_value=32, max_value=256, step=32
                        ),
                        kernel_size=hp.Choice(f"kernel_size_{i}", [3, 5, 7]),
                        activation="relu",
                    )
                )
                model.add(
                    Dropout(
                        hp.Float(
                            f"dropout_rate_conv_{i}",
                            min_value=0.1,
                            max_value=0.5,
                            step=0.1,
                        )
                    )
                )
            model.add(GlobalMaxPooling1D())
            model.add(
                Dense(
                    units=hp.Int("hidden_units", min_value=32, max_value=256, step=32),
                    activation="relu",
                )
            )
            model.add(
                Dropout(
                    hp.Float(
                        "dropout_rate_dense", min_value=0.1, max_value=0.5, step=0.1
                    )
                )
            )
            model.add(Dense(units=1, activation="sigmoid"))
            model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(
                    learning_rate=hp.Float(
                        "learning_rate",
                        min_value=1e-4,
                        max_value=1e-2,
                        sampling="LOG",
                        default=1e-3,
                    )
                ),
                metrics=["accuracy"],
            )
            return model

    hypermodel = IMDBHyperModel()

    tuner = Hyperband(
        hypermodel,
        objective="val_accuracy",
        max_epochs=10,
        hyperband_iterations=2,
        directory="hyperband",
        project_name="imdb_reviews",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    tuner.search(
        x_train_padded,
        y_train,
        epochs=epochs,
        validation_data=(x_test_padded, y_test),
        batch_size=batch_size,
        callbacks=[early_stopping],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        x_train_padded,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test_padded, y_test),
        callbacks=[early_stopping],
    )

    score = model.evaluate(x_test_padded, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    model.save("model.keras")
