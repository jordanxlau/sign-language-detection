import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

X = pd.read_csv("X.csv")
y = pd.read_csv("y.csv")

# Randomly shuffle and split data into training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)

# Convert to Tensorflow Tensor
X_train, X_test, y_train, y_test = tf.constant(X_train), tf.constant(X_test), tf.constant(y_train), tf.constant(y_test)

# Create a Neural Network
model = keras.Sequential([
  keras.layers.InputLayer(shape=(63,)),
  keras.layers.Flatten(),
  keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
  keras.layers.Dropout(0.10),
  keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
  keras.layers.Dropout(0.10),
  keras.layers.Dense(units=5, activation="softmax")
])

# Compile the model
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
)

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.summary()

# Save
model.save("model-fivesymbols.keras", overwrite=True)