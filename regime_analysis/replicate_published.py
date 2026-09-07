"""
replicate_published.py
Reproduce the ORIGINAL evaluation protocol, so that any change in the numbers can
be attributed to the protocol rather than to this reimplementation.

The published pipeline did three things that the corrected protocol changes:

1. split_by_date=False, i.e. sklearn train_test_split on overlapping sequence
   windows. Neighbouring windows share 19 of 20 input days, so near-duplicates
   land on both sides of the split.
2. The LSTM was evaluated on the whole test set, but the KAN script evaluated on
   x_test[-STEPS_TO_PREDICT:] only, i.e. the last h rows of the shuffled test
   set. At h=1 that is a single sample.
3. No fixed seed, so every rerun draws a different split and a different
   initialisation.

Running this file prints the numbers this protocol produces. If they land in the
range published in Tables 4-6 and quoted in the text, the reimplementation is
sound and the difference in the corrected results is caused by the protocol.

    python replicate_published.py
"""

import os

import numpy as np
from sklearn.model_selection import train_test_split

import regime_lib as rl

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPEATS = 3


def lstm_original(x_train, y_train, x_test, y_test, units, activation, horizon, seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.layers.Input(shape=x_train.shape[1:])]
    for index in range(4):
        layers.append(tf.keras.layers.LSTM(units, return_sequences=index < 3))
        layers.append(tf.keras.layers.Dropout(0.2))
    layers.append(tf.keras.layers.Dense(horizon, activation=activation))
    model = tf.keras.Sequential(layers)
    model.compile(loss="mean_squared_error", optimizer="RMSprop",
                  metrics=["mean_squared_error"])
    model.fit(x_train, y_train, epochs=25, batch_size=30, verbose=0)

    train_rmse = rl.rmse(model.predict(x_train, verbose=0), y_train)
    test_rmse = rl.rmse(model.predict(x_test, verbose=0), y_test)
    return train_rmse, test_rmse


def kan_original(x_train, y_train, x_test, y_test, horizon, seed):
    """Note the subset step: the published KAN script kept only the last h rows."""
    import tensorflow as tf

    from kan_layer import build_kan, fit_lbfgs

    tf.keras.utils.set_random_seed(seed)
    flat_train = x_train.reshape(len(x_train), -1)

    subset_x = x_test[-horizon:]
    subset_y = y_test[-horizon:]
    flat_subset = subset_x.reshape(len(subset_x), -1)

    hidden = max(3, 3 * (len(flat_train) // 50))
    model = build_kan(flat_train.shape[1], hidden, horizon, seed=seed)
    fit_lbfgs(model, flat_train, y_train, max_iterations=10)

    train_rmse = rl.rmse(model.predict(flat_train, verbose=0), y_train)
    test_rmse = rl.rmse(model.predict(flat_subset, verbose=0), subset_y)
    return train_rmse, test_rmse, len(subset_y)


def main():
    dataframe = rl.load_prices()
    print("protocol: shuffled train_test_split, no fixed split seed, "
          "KAN scored on the last h test rows only")
    print()

    for horizon, units, activation in [(1, 100, "linear"), (2, 10, "tanh"),
                                       (100, 10, "tanh")]:
        x_all, y_all, _ = rl.build_sequences(dataframe, horizon)
        lstm_scores, kan_scores = [], []
        subset_size = None
        for repeat in range(REPEATS):
            x_train, x_test, y_train, y_test = train_test_split(
                x_all, y_all, test_size=0.2
            )
            _, lstm_test = lstm_original(
                x_train, y_train, x_test, y_test, units, activation, horizon,
                seed=repeat
            )
            _, kan_test, subset_size = kan_original(
                x_train, y_train, x_test, y_test, horizon, seed=repeat
            )
            lstm_scores.append(lstm_test)
            kan_scores.append(kan_test)

        print(
            "h=%-3d LSTM(%du-%s) test RMSE  min %.4f  mean %.4f  max %.4f"
            % (horizon, units, activation, min(lstm_scores), float(np.mean(lstm_scores)),
               max(lstm_scores))
        )
        print(
            "      KAN            test RMSE  min %.4f  mean %.4f  max %.4f "
            "(scored on %d sample%s)"
            % (min(kan_scores), float(np.mean(kan_scores)), max(kan_scores),
               subset_size, "" if subset_size == 1 else "s")
        )
        print()


if __name__ == "__main__":
    main()
