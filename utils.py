import keras


def pad_sequences(x_train, x_test, maxlen):
    return keras.utils.pad_sequences(x_train, maxlen=maxlen), keras.utils.pad_sequences(
        x_test, maxlen=maxlen
    )
