from preprocessing import X_train, X_test, y_train, y_test
import keras
from functools import partial

# The architecture of this Neural Network was been borrowed from my CSI 4106 lecture notes

# Create a Convolutional Neural Network
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")

model = keras.Sequential([
  DefaultConv2D(filters=64, kernel_size=3, input_shape=(28, 28, 1)),
  keras.layers.MaxPool2D(),
  DefaultConv2D(filters=128),
  DefaultConv2D(filters=128),
  keras.layers.MaxPool2D(),
  DefaultConv2D(filters=256),
  DefaultConv2D(filters=256),
  keras.layers.MaxPool2D(),
  keras.layers.Flatten(),
  keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
  keras.layers.Dropout(0.10),
  keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
  keras.layers.Dropout(0.10),
  keras.layers.Dense(units=26, activation="softmax")
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
model.save("model-wholealphabet.keras", overwrite=True)