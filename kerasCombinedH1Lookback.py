import keras
import math
import os
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error, precision_score, f1_score, recall_score
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
import time

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#creates dataset


output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_dataset2(dataset, lookback):
  dataset = np.array(dataset)
  X, y = [], []
  for i in range(len(dataset)-lookback-1):
    if i+lookback > len(dataset):
        continue
    feature = dataset[i:i+lookback]
    target = dataset[i+lookback]
    X.append(feature)
    y.append(target)

  return X,y

#opens csv file and gets data from it
csv_list = ['combined_H1_edited.csv']
df_occupancy = pd.DataFrame()  # main data storage
for file in csv_list:
    df_temp = pd.read_csv(file)
    df_occupancy = pd.concat([df_occupancy, df_temp], ignore_index=True)

#processes the data to only get the occupancies from each row
print("File name", csv_list)
print("Occupancy data before dropping NA data", df_occupancy.shape)
df_occupancy = df_occupancy.dropna(how='any', axis=0)
print("Occupancy data after dropping NA data", df_occupancy.shape)


data = df_occupancy['number']

#only gets every 7 rows to minimize data
samples = 7
data = data.iloc[::samples]

print("y shape", data.shape)

#plots dist.png to show data distribution
pyplot.figure(figsize=(10, 6))
pyplot.hist(data, bins=20, color='skyblue', edgecolor='black')
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
"""
  timeseries_len = y.shape[0]
train_size = int(timeseries_len * 0.7)
test_size = timeseries_len - train_size
train, test = y[:train_size], y[train_size:]
"""
lookback_values = [30]
lookback_accuracy = []
lookback_time = []
for lookback in lookback_values:
    X, y = create_dataset2(data, lookback=lookback)
    #one dataset call then split after, make sure both test and train have all outputs that they should have
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state=2024)
  
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
        head_size = 32,
        num_heads=8,
        ff_dim=32,
        num_transformer_blocks=4,
        mlp_units=[56, 16],
        mlp_dropout=0.3,
        dropout=0.25, 
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    #model.summary()

    start_time = time.time()
    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    epochs = 20

    #change epochs to make the model learn longer and hopefully get more accurate
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs, #PARAMETER
        batch_size=64, #PARAMETER
        callbacks=callbacks,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed_time))
    lookback_time.append(elapsed_time_str)
    #runs model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    accuracy = round(accuracy, 4)
    lookback_accuracy.append(accuracy)

    # Step 1: Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate additional metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_classes))
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    # Step 2: Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # Step 3: Plot the confusion matrix
    pyplot.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    pyplot.xlabel('Predicted Labels')
    pyplot.ylabel('True Labels')
    pyplot.title(f'Confusion Matrix (Epochs = {epochs}) (Runtime = {elapsed_time_str}) (Accuracy = {accuracy})\n'
    f'RMSE: {rmse:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}')

    pyplot.savefig(os.path.join(output_dir, f'H1ConfusionMatrix_{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}.png'))
"""
    pyplot.show()
    pyplot.figure(figsize=(14, 7))
    pyplot.plot(y_test, label='Actual Values', color='blue')
    pyplot.plot(y_pred_classes, label='Predicted Values', color='red', alpha=0.7)
    pyplot.xlabel('Sample Index')
    pyplot.ylabel('Class Labels')
    pyplot.title('Actual vs Predicted Values')
    pyplot.legend()
    actual_vs_pred_filename = os.path.join(output_dir, f'H1actualVsPred_{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}.png')
    pyplot.savefig(actual_vs_pred_filename)
    pyplot.show
"""
for i in range(0, len(lookback_values)):
    print(f'Lookback: {lookback_values[i]}, Accuracy: {lookback_accuracy[i]}, Runtime: {lookback_time[i]}')
