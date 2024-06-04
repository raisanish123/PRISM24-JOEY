import keras
import math
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def create_dataset2(dataset, lookback):
  dataset = np.array(dataset)
  X, y = [], []
  for i in range(len(dataset)-lookback-1):
    feature = dataset[i:i+lookback]
    target = dataset[i+lookback]
    X.append(feature)
    y.append(target)

  return X,y

csv_list = ['combined_H1_edited.csv']
df_occupancy = pd.DataFrame()  # main data storage
for file in csv_list:
    df_temp = pd.read_csv(file)
    df_occupancy = pd.concat([df_occupancy, df_temp], ignore_index=True)

# Preprocess the data
print("File name", csv_list)
print("Occupancy data before dropping NA data", df_occupancy.shape)
df_occupancy = df_occupancy.dropna(how='any', axis=0)
print("Occupancy data after dropping NA data", df_occupancy.shape)

# Separate features and target
#X = df_occupancy.drop(['date', 'occupied', 'number'], axis=1)

y = df_occupancy['number']

# Select every sample defined row
samples = 7
y = y.iloc[::samples]
#print("X shape", X.shape)
print("y shape", y.shape)

# MinMax scaling
#scaler = MinMaxScaler(feature_range = (-1,1)).fit(X)
#X = scaler.transform(X)
# Standard scaling
"""
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
print("X shape after scaling", X.shape)
"""
# Split the data into training and testing sets
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2024)
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)
"""
y = y[:10000]
timeseries_len = y.shape[0]
train_size = int(timeseries_len * 0.7)
test_size = timeseries_len - train_size
train, test = y[:train_size], y[test_size:]
lookback = 100
X_train, y_train = create_dataset2(train, lookback=lookback)
X_test, y_test = create_dataset2(test, lookback=lookback)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# Reshape the data to add a channel dimension
X_train = X_train.reshape((X_train.shape[0], lookback, 1))
X_test = X_test.reshape((X_test.shape[0], lookback, 1))
print('XTEAIN SHAPW', X_train.shape)
print('XTEST SHAPE', X_test.shape)
print('Ytrain SHAPW', len(y_train))
print('Ytest SHAPE', len(y_test))
# Get the number of classes
n_classes = len(np.unique(y_train))+1 #saying 1 less class

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

input_shape = X_train.shape[1:]
print('input shape', input_shape)
model = build_model(
    input_shape,
    head_size=512,
    num_heads=8,
    ff_dim=256,
    num_transformer_blocks=4,
    mlp_units=[256, 128, 56],
    mlp_dropout=0.4,
    dropout=0.25,
)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(X_test, y_test, verbose=1)
