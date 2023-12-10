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

# Visualize the first training image and its label
plt.imshow(x_train[0].reshape(28, 28), cmap='gray') # If you used CNN, reshape back to 2D for viewing
plt.title(f'Label: {y_train[0]}')
plt.show()

# Optionally, visualize the results of predictions on test data
# ... (visualization code from the previous step)
