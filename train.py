import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

dataset_landmarked = pd.read_csv("dataset_landmarked-29.csv")

X = dataset_landmarked.iloc[:,1:]
y = dataset_landmarked.iloc[:,0]

# Randomly shuffle and split data into training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)

# Convert to Tensorflow Tensor
X_train, X_test, y_train, y_test = tf.constant(X_train), tf.constant(X_test), tf.constant(y_train), tf.constant(y_test)

# Create a Neural Network
model = keras.Sequential([
  keras.layers.InputLayer(shape=(63,)), # 21 landmarks * 3 coordinates
  keras.layers.Flatten(),
  keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
  keras.layers.Dropout(0.10),
  keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
  keras.layers.Dropout(0.10),
  keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
  keras.layers.Dropout(0.10),
  keras.layers.Dense(units=29, activation="softmax")
])

# Compile the model
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

# Train the model
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

plt.title("Model Training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(history.history['accuracy'], color=(0.9, 0, 0), linestyle='dashed')
plt.plot(history.history['val_accuracy'], color=(0.9, 0, 0.4))
plt.show()

# Save
model.save("hand-gestures-29.keras", overwrite=True)