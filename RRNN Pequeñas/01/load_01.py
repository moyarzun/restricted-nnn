import numpy as np
from keras.models import load_model
from keras.utils import np_utils
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import zmq, pickle, sys, argparse

filenames = '01'
port_in = '5001'
port_out = '5002'

# ZeroMQ Context
context = zmq.Context()

# Define the socket using the "Context"
sock = context.socket(zmq.REP)

## Gets IP and PORT from command line and parses them
ConnectionInfo = argparse.ArgumentParser()
ConnectionInfo.add_argument("-i",  default='127.0.0.1')
ConnectionInfo.add_argument("-o",  default='127.0.0.1')
ConnectionInfoParsed = ConnectionInfo.parse_args()

# Saves the parsed IP and Port
ip_in = ConnectionInfoParsed.i
ip_out = ConnectionInfoParsed.o

try:
    sock.connect('tcp://'+ip_in+':'+port_in)
except:
    print('Usage: python load_'+filenames+'.py -i <valid input ip address> -o <valid output ip address>')

# Fijar semilla para reproducir experimento
seed = 2141
np.random.seed(seed)

# Run a simple "Echo" server
print('Listening to tcp://'+ip_in+':'+port_in+'...')
X_message = sock.recv()
print('Receiving data...')
X_test = pickle.loads(X_message)
sock.send(pickle.dumps(X_message))
# print(X_message)
y_message = sock.recv()
y_test = pickle.loads(y_message)
sock.send(pickle.dumps(y_message))
print('Data received. Starting classification...')

X_send = X_test
y_send = y_test

# Descargar dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

###############################
# This is the key... order is important!
y_test[y_test<=0]=2
y_test[y_test==1]=0
y_test[y_test==2]=1
y_test[y_test>=3]=2

print(np.unique(y_test))
# [0 1 2]
###############################

# let's print the shape before we reshape and normalize
# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# building the input vector from the 28x28 pixels
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalizing the data to help with the training
# X_train /= 255
X_test /= 255

# print the final input shape ready for training
# print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

# one-hot encoding using keras' numpy-related utilities
n_classes = 3
# print("Shape before one-hot encoding: ", y_train.shape)
# Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
# print("Shape after one-hot encoding: ", Y_train.shape)

# Cargar modelo preguardado
model = load_model(filenames + '_model.h5')
model.load_weights(filenames + '_tensors.h5')

loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

# load the model and create predictions on the test set
predicted_classes = model.predict_classes(X_test)

# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                        y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(predicted_classes[incorrect],
                                       y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

figure_evaluation

plt.savefig(filenames+'_load_pred.png')
print("Predictions saved as '" + filenames + "_load_pred.png'.")
print("Sending data to the next node at tcp://"+ip_out":"port_out"...")
# Preparing ZeroMQ context for the next node...
sock = context.socket(zmq.REQ)
sock.bind('tcp://'+ip_out+':'+port_out)
sock.send(pickle.dumps(X_send))
X_answer = sock.recv()
sock.send(pickle.dumps(y_send))
y_answer = sock.recv()
# sock.send(pickle.dumps(y_message))
print('Done')
