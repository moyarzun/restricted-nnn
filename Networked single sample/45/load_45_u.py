import numpy as np
from keras.models import load_model
from keras.utils import np_utils
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import zmq, pickle, sys, argparse, copy

filenames = '45_u'
port = '5000'
port_out = '5001'

# ZeroMQ Context
context = zmq.Context()

# Define the socket using the "Context"
sock = context.socket(zmq.REP)

## Gets IP and PORT from command line and parses them
ConnectionInfo = argparse.ArgumentParser()
ConnectionInfo.add_argument("-i",  default='127.0.0.1')
ConnectionInfo.add_argument("-o",  default='0.0.0.0')
ConnectionInfoParsed = ConnectionInfo.parse_args()

# Saves the parsed IP and Port
ip_in = ConnectionInfoParsed.i
ip_out = ConnectionInfoParsed.o

try:
    sock.connect('tcp://'+ip_in+':'+port)
except:
    print('Usage: python load_'+filenames+'.py -i <valid input ip address> -o <valid output ip address>')

# Fijar semilla para reproducir experimento
seed = 2141
np.random.seed(seed)

# Run a simple "Echo" server
print('Listening to tcp://'+ip_in+':'+port+'...')
X_message = sock.recv()
print('Receiving data...')
X_test = pickle.loads(X_message)
sock.send(pickle.dumps(X_message))
sock.close()
print('Data received. Starting classification...')
message = X_test

# building the input vector from the 28x28 pixels
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalizing the data to help with the training
X_test /= 255

# Cargar modelo preguardado
model = load_model(filenames + '_model.h5')
model.load_weights(filenames + '_tensors.h5')

# load the model and create predictions on the test set
predicted_classes = model.predict_classes(X_test)

print()
print("Value predicted: ", predicted_classes)
print()
if predicted_classes == 2:
    print("Predicted class: 'other'...")
    print("Continuing classification at next node...")
    # ZeroMQ Context
    context = zmq.Context()
    # Preparing ZeroMQ context for the next node...
    sock = context.socket(zmq.REQ)
    sock.connect('tcp://'+ip_out+':'+port)
    sock.send(pickle.dumps(message))
    X_answer = sock.recv()
    print('Data sent to the next node.')
    sock.close()
else:
    print("predicted class: ", predicted_classes)
    sock = context.socket(zmq.REQ)
    sock.connect('tcp://'+ip_out+':'+port_out)
    sock.send(pickle.dumps(predicted_classes+4))
    end_string = sock.recv()

print('Done!')
