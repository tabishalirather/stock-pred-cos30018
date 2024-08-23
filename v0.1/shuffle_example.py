import numpy as np
import pandas as pd

# Creating the example stock market data
dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
features = np.array([
    [100, 101, 102],
    [103, 104, 105],
    [106, 107, 108],
    [109, 110, 111],
    [112, 113, 114]
])
labels = np.array([1, 0, 1, 0, 1])  # Example labels for binary classification

# Original data
original_data = pd.DataFrame(features, columns=['Open', 'High', 'Low'], index=dates)
original_data['Label'] = labels


# Shuffling using shuffle_in_unison method
def shuffle_in_unison(a, b, c):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    np.random.set_state(state)
    np.random.shuffle(c)


# Convert dates to list for shuffling
dates_unison = dates.to_list()
X_train_unison = features.copy()
y_train_unison = labels.copy()
shuffle_in_unison(X_train_unison, y_train_unison, dates_unison)


# Shuffling using block-based method (block size of 2)
def shuffle_blocks(a, b, c, block_size):
    # Split data into blocks
    num_blocks = len(a) // block_size
    a_blocks = np.array_split(a, num_blocks)
    b_blocks = np.array_split(b, num_blocks)
    c_blocks = np.array_split(c, num_blocks)

    # Combine blocks into tuples
    blocks = list(zip(a_blocks, b_blocks, c_blocks))

    # Shuffle the blocks
    np.random.shuffle(blocks)

    # Unzip the blocks back into separate arrays
    a_shuffled, b_shuffled, c_shuffled = zip(*blocks)

    # Concatenate blocks back together
    return np.concatenate(a_shuffled), np.concatenate(b_shuffled), np.concatenate(c_shuffled)


# Convert dates to list for shuffling
dates_blocks = dates.to_list()
X_train_blocks = features.copy()
y_train_blocks = labels.copy()

# Shuffle blocks (block size of 2)
block_size = 2
X_train_blocks_shuffled, y_train_blocks_shuffled, dates_blocks_shuffled = shuffle_blocks(
    X_train_blocks, y_train_blocks, dates_blocks, block_size
)

# Convert back to DatetimeIndex after shuffling
dates_unison = pd.to_datetime(dates_unison)
dates_blocks_shuffled = pd.to_datetime(dates_blocks_shuffled)

# Convert to DataFrame for better visualization
unison_shuffled_df = pd.DataFrame(X_train_unison, columns=['Open', 'High', 'Low'], index=dates_unison)
unison_shuffled_df['Label'] = y_train_unison

blocks_shuffled_df = pd.DataFrame(X_train_blocks_shuffled, columns=['Open', 'High', 'Low'], index=dates_blocks_shuffled)
blocks_shuffled_df['Label'] = y_train_blocks_shuffled

# Displaying results
print("Original Data:")
print(original_data)
print("\nUnison Shuffled Data:")
print(unison_shuffled_df)
print("\nBlock Shuffled Data:")
print(blocks_shuffled_df)
