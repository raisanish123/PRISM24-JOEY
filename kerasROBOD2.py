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
from matplotlib import pyplot
import seaborn as sns
from sklearn.metrics import confusion_matrix


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#creates dataset
def create_dataset2(dataset, lookback):
  dataset = np.array(dataset)
  X, y = [], []
  for i in range(len(dataset)-lookback-1):
    feature = dataset[i:i+lookback]
    target = dataset[i+lookback]
    X.append(feature)
    y.append(target)

  return X,y

#opens csv file and gets data from it
csv_list = ['combined_Room1.csv']
df_occupancy = pd.DataFrame()  # main data storage
for file in csv_list:
    df_temp = pd.read_csv(file)
    df_occupancy = pd.concat([df_occupancy, df_temp], ignore_index=True)

#processes the data to only get the occupancies from each row
print("File name", csv_list)
print("Occupancy data before dropping NA data", df_occupancy.shape)
df_occupancy = df_occupancy.dropna(how='any', axis=0)
df_occupancy = df_occupancy[df_occupancy['occupant_count [number]'] != 0]

print("Occupancy data after dropping NA data", df_occupancy.shape)


y = df_occupancy['occupant_count [number]']

#only gets every 7 rows to minimize data
samples = 7
y = y.iloc[::samples]

print("y shape", y.shape)

#plots dist.png to show data distribution
pyplot.figure(figsize=(10, 6))
pyplot.hist(y, bins=20, color='skyblue', edgecolor='black')
pyplot.xlabel('Occupant Count')
pyplot.ylabel('Frequency')
pyplot.title('Distribution of Occupant Count')
pyplot.grid(True)
pyplot.show()
pyplot.savefig('dist.png')
"""
# MinMax scaling
#scaler = MinMaxScaler(feature_range = (-1,1)).fit(X)
#X = scaler.transform(X)
# Standard scaling

scaler = StandardScaler().fit(X)
X = scaler.transform(X)
print("X shape after scaling", X.shape)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2024)
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)
"""
#y = y[:5000]
#code that takes the data and splits it into testing and training dataset to run model on
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

# Reshape the data
X_train = X_train.reshape((X_train.shape[0], lookback, 1))
X_test = X_test.reshape((X_test.shape[0], lookback, 1))

print('XTRAIN SHAPE', X_train.shape)
print('XTEST SHAPE', X_test.shape)
print('YTRAIN SHAPE', len(y_train))
print('YTEST SHAPE', len(y_test))

# Get the number of classes, aka the number of unique occupancies that exist
#n_classes = len(np.unique(y_train))+1 #saying 1 less class
n_classes = int((np.max(y_train)) + 1)
print("NUMBER CLASSES", n_classes)
print(np.unique(y_train))
print(np.unique(y_test))



#TRANSFORMER MODEL
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

#Here we start to build the model
input_shape = X_train.shape[1:]
print('input shape', input_shape)
#CHANGE THE PARAMETERS TO GET DIFFERENT RESULTS
model = build_model(
    input_shape,
    head_size=256, #PARAMETER
    num_heads=8, #PARAMETER
    ff_dim=256, #PARAMETER
    num_transformer_blocks=4, #PARAMETER
    mlp_units=[128, 56], #PARAMETER can change to somthing like [256, 128, 56] or [56]
    mlp_dropout=0.4, #PARAMETER
    dropout=0.25, #PARAMETER
)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

#change epochs to make the model learn longer and hopefully get more accurate
model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=5, #PARAMETER
    batch_size=64, #PARAMETER
    callbacks=callbacks,
)

#runs model
model.evaluate(X_test, y_test, verbose=1)


# Step 1: Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Step 2: Plot the actual vs predicted values using a line plot
pyplot.figure(figsize=(14, 7))

# Plot actual values
pyplot.plot(y_test, label='Actual Values', color='blue')

# Plot predicted values
pyplot.plot(y_pred_classes, label='Predicted Values', color='red', alpha=0.7)

# Labels and title
pyplot.xlabel('Sample Index')
pyplot.ylabel('Class Labels')
pyplot.title('Actual vs Predicted Values')
pyplot.legend()
pyplot.savefig('actualVsPred.png')

# Display the plot
pyplot.show()

  # Step 1: Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Step 2: Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Step 3: Plot the confusion matrix
pyplot.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
pyplot.xlabel('Predicted Labels')
pyplot.ylabel('True Labels')
pyplot.title('Confusion Matrix')
pyplot.savefig('ConfusionMatrix.png')
pyplot.show()

print('Go into files find the directory you are working in then opin actualVsPred.png, ConfusionMatrix.png, and dist.png')
