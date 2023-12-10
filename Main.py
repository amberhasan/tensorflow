import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training data shape:", x_train.shape)  # Should be (60000, 28, 28)
print("Test data shape:", x_test.shape)      # Should be (10000, 28, 28)
print("Training label shape:", y_train.shape) # Should be (60000,)
print("Test label shape:", y_test.shape)      # Should be (10000,)


plt.imshow(x_train[0], cmap='gray')
plt.title('Label: ' + str(y_train[0]))
plt.show()
