import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils
from datetime import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import zmq, sys, pickle, argparse, copy, math

filenames = '01_u'
port = '5000'
port_end = '5001'

# Fijar semilla para reproducir experimento
seed = 2141
np.random.seed(seed)

# Descargar dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## Gets IP and PORT from command line and parses them
ConnectionInfo = argparse.ArgumentParser()
ConnectionInfo.add_argument("-i",  default='127.0.0.1')
ConnectionInfo.add_argument("-o",  default='0.0.0.0')
ConnectionInfo.add_argument("-c",  default=math.floor(np.random.random() * 1000))
ConnectionInfoParsed = ConnectionInfo.parse_args()

# Saves the parsed IP and Port
ip_in = ConnectionInfoParsed.i
ip_out = ConnectionInfoParsed.o
sample = int(ConnectionInfoParsed.c)

print("MNIST Sample", int(sample))
foo = X_test[sample]
print("Expected class: ", y_test[sample])

#Inicia medición de tiempo
start = datetime.now()

print('Image preprocessing...')
foo = np.expand_dims(foo, axis=0)
message = foo

# building the input vector from the 28x28 pixels
foo = foo.reshape(foo.shape[0], 1, 28, 28).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalizing the data to help with the training
foo /= 255

print('Preprocessing done.')
print('Loading model and tensors...')

# Cargar modelo preguardado
model = load_model(filenames + '_model.h5')
model.load_weights(filenames + '_tensors.h5')

print('Model loading done.')
print('Classifying...')

# load the model and create predictions on the test set
class_start = datetime.now()
predicted_classes = model.predict_classes(foo)
class_end = datetime.now() - class_start

print('---------------------------')
print("Value predicted: ", predicted_classes)
print('Classification done in (hh:mm:ss.ms) {}'.format(class_end))
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
    personal = datetime.now() - start
    print('Node classification time (hh:mm:ss.ms) {}'.format(personal))
    print('Data sent. Waiting for classification...')
    sock.close()

    # Espera hasta que concluya la clasificación
    sock = context.socket(zmq.REQ)
    sock.bind('tcp://0.0.0.0:'+port_end)
    for x in range (1, 6):
        sock.send_string('ack')
        result = sock.recv()
        result = pickle.loads(result)
        if result == -1:
            print("Node "+ str(x) +" couldn't classify your sample.")
        else:
            print("Node "+ str(x) +" predicted class: ", result)
            break
    sock.close()
else:
    print("Predicted class: ", predicted_classes)

total = datetime.now() - start
print('Network processing time (hh:mm:ss.ms) {}'.format(total))
print('Done!')
