import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
# Load your dataset (e.g., MNIST)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
model = keras.Sequential([
layers.Flatten(input_shape=(28, 28)), # Flatten the input image
layers.Dense(128, activation='relu'), # First hidden layer with ReLU activation
layers.Dense(10, activation='softmax') # Output layer with softmax activation
])
model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)
model.fit(x_train, y_train, epochs=20, batch_size=1, validation_data=(x_test, y_test))
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
model = keras.Sequential([
layers.Flatten(input_shape=(28, 28)),
layers.Dense(256, activation='relu'), # First hidden layer
layers.Dense(128, activation='relu'), # Second hidden layer
layers.Dense(10, activation='softmax') # Output layer
])
