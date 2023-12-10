import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# If using a CNN, add a channel dimension, else skip this step for a simple dense network
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # Flatten layer for CNN input
  tf.keras.layers.Dense(128, activation='relu'),    # First dense layer
  tf.keras.layers.Dropout(0.2),                     # Dropout for regularization
  tf.keras.layers.Dense(10, activation='softmax')   # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5) # has the logs with the epoch. 

# Evaluate the model
model.evaluate(x_test, y_test)

# Make predictions
predictions = model.predict(x_test)

# Choose an image index
index = 0  # for example, the first image in the test set

# Predict the class (digit) for the chosen image
predicted_class = np.argmax(predictions[index])

# Visualize the chosen test image and its prediction
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title(f'Actual Label: {y_test[index]}, Predicted Label: {predicted_class}')
plt.show()
