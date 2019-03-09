import cv2
import numpy as np
import glob
import sys

print ("OpenCV:",  cv2.__version__)
print ("Numpy : ", np.__version__)
print ("Python:",  sys.version)

# load training data
dim = 28800
X = np.empty((1, dim))
y = np.empty((1, 4))
training_data = glob.glob('training_data/*.npz')

for single_npz in training_data:
    with np.load(single_npz) as data:
        train = data['train']
        train_labels = data['train_labels']
#print(train)
#print(train_labels)

    X = np.vstack((X, train))
    y = np.vstack((y, train_labels))

print ('Image array shape: ', X.shape)
print ('Label array shape: ', y.shape)

# create model
#model = cv2.ml.ANN_MLP_create()
#layer_sizes = np.int32([dim, 32, 32, 4])
#model.setLayerSizes(layer_sizes)
#model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
#model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
#model.setBackpropWeightScale(0.001)
#model.setBackpropMomentumScale(0.1)
#model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))

# train
#model.train(np.float32(X), cv2.ml.ROW_SAMPLE, np.float32(y))

# predict
#ret, resp = model.predict(X)
#prediction = resp.argmax(-1)
#true_labels = y.argmax(-1)

#train_rate = np.mean(prediction == true_labels)
#print (prediction)
#print ('Train accuracy: ', "{0:.2f}%".format(train_rate * 100))'''
