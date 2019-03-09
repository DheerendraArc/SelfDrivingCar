import cv2
import numpy as np
import glob
import sys

print('OpenCV: ', cv2.__version__)
print('Numpy: ', np.__version__)
print('Python: ', sys.version)

# Loading the Training Data
dim = 38400
X = np.empty((1, dim))
y = np.empty((1, 4))

print(X)
print(y)
print(X.shape)
print(y.shape)

training_data = glob.glob('training_data/*.npz')

for single_npz in training_data:
    with np.load(single_npz) as data:
          train = data['train']
          train_labels = data['train_labels']
#print(train)
#print(train_labels)

#print(train.shape)
#print(train_labels.shape)
X = np.vstack((X, train))
y = np.vstack((y, train_labels))
#print(X)
#print(y)

#print(X.shape)
#print(y.shape)
print('Image array shape: ', X.shape)
print('Label array shape: ', y.shape)

# Create Model

model = cv2.ml.ANN_MLP_create()

layer_sizes = np.int32([dim, 32, 32, 4])
model.setLayerSizes(layer_sizes)
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
model.setBackpropWeightScale(0.001)

# Train Model

model.train(np.float32(X), cv2.ml.ROW_SAMPLE, np.float32(y))

# Predict

ret, resp = model.predict(X)
prediction = resp.argmax(-1)
print('Prediction: ', prediction)
true_labels = y.argmax(-1)
print('true_labels: ', true_labels)

train_rate = np.mean(prediction == true_labels)
print(prediction)

print('Train accuracy: ','{0:.2f}%'.format(train_rate * 100))

model.save('mlp_xml/mlp.xml')
