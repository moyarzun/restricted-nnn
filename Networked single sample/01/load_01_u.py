import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import zmq, sys, pickle, argparse, copy, math

filenames = '01_u'
port = '5000'
port_end = '5001'

## Gets IP and PORT from command line and parses them
ConnectionInfo = argparse.ArgumentParser()
ConnectionInfo.add_argument("-i",  default='127.0.0.1')
ConnectionInfo.add_argument("-o",  default='0.0.0.0')
ConnectionInfoParsed = ConnectionInfo.parse_args()

# Saves the parsed IP and Port
ip_in = ConnectionInfoParsed.i
ip_out = ConnectionInfoParsed.o

# Fijar semilla para reproducir experimento
seed = 2141
np.random.seed(seed)
sample = math.floor(np.random.random() * 1000)
print("sample", sample)

# Descargar dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

foo = X_test[1]
print("Class: ", y_test[1])
foo = np.expand_dims(foo, axis=0)
message = foo

# building the input vector from the 28x28 pixels
foo = foo.reshape(foo.shape[0], 1, 28, 28).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalizing the data to help with the training
foo /= 255

# print the final input shape ready for training
print("Foo matrix shape", foo.shape)

# Cargar modelo preguardado
model = load_model(filenames + '_model.h5')
model.load_weights(filenames + '_tensors.h5')

# load the model and create predictions on the test set
predicted_classes = model.predict_classes(foo)

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
    sock.bind('tcp://0.0.0.0:'+port)
    sock.send(pickle.dumps(message))
    X_answer = sock.recv()
    print('Data sent. Waiting for classification.')
    sock.close()

    # Espera hasta que concluya la clasificación
    sock = context.socket(zmq.REQ)
    sock.bind('tcp://0.0.0.0:'+port_end)
    end_classif = sock.recv()
    sock.send_string('ack')
    sock.close()
    end_result = pickle.loads(end_classif)
    if end_result == -1:
        print("Network couldn't classify tour sample. Sorry! =(")
    else:
        print("Predicted class: ", end_result)
else:
    print("Predicted class: ", predicted_classes)

print('Done!')
