import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Fixed seed for experiment replayability
seed = 2141
np.random.seed(seed)

filenames = 'little'
n_classes = 3

# Dataset download (If not local, from Internet)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

###############################
# Re-tagging the dataset for the specific classes we want to train
# 0 -> Class 1
# 1 -> Class 2
# 2 -> 'Others'
y_train[y_train==0]=0
y_train[y_train==1]=1
y_train[y_train>=2]=2

y_test[y_test==0]=0
y_test[y_test==1]=1
y_test[y_test>=2]=2
###############################

# Building the input vector from the 28x28 pixels
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Normalizing the data
X_train /= 255
X_test /= 255

# One-hot encoding
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

# Model design
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Model building
model = baseline_model()

# Model compiling
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Model training and metrics capture
history = model.fit(X_train, Y_train,
          batch_size=200, epochs=5,
          verbose=1,
          validation_data=(X_test, Y_test))

# Saving model
model.save(filenames + '_model.h5')
# Saving weights
model.save_weights(filenames + '_tensors.h5')
print('Model saved.')

# Plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

plt.savefig(filenames + '_build_met.png')
print("Metrics saved as " + filenames + "'_build_met.png'.")

loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

# load the model and create predictions on the test set
predicted_classes = model.predict_classes(X_test)

# See which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

# Adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# Plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                        y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# Plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(predicted_classes[incorrect],
                                       y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

figure_evaluation

plt.savefig(filenames + '_build_pred.png')
print("Predictions saved as '" + filenames + "_build_pred.png'.")
