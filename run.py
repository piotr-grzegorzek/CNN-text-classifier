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
from keras.src.utils import pad_sequences  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
from keras_tuner import HyperModel
from keras_tuner.tuners import Hyperband
from sklearn.model_selection import train_test_split  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

max_features = 20000
max_len = 100
batch_size = 32
epochs = 10

data = pd.read_csv(
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    header=None,
)
data.columns = ["Category", "Title", "Description"]
data["Text"] = data["Title"] + " " + data["Description"]
x_data = data["Text"].astype(str)
y_data = data["Category"] - 1

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_data)
x_data_seq = tokenizer.texts_to_sequences(x_data)
x_data_padded = pad_sequences(x_data_seq, maxlen=max_len)

x_train, x_test, y_train, y_test = train_test_split(
    x_data_padded, y_data, test_size=0.2, random_state=42
)

if __name__ == "__main__":

    class AGNewsHyperModel(HyperModel):
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
            model.add(Dense(units=4, activation="softmax"))
            model.compile(
                loss="sparse_categorical_crossentropy",
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

    hypermodel = AGNewsHyperModel()

    tuner = Hyperband(
        hypermodel,
        objective="val_accuracy",
        max_epochs=10,
        hyperband_iterations=2,
        directory="hyperband",
        project_name="ag_news",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    tuner.search(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        callbacks=[early_stopping],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping],
    )

    score = model.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    model.save("model.keras")

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
