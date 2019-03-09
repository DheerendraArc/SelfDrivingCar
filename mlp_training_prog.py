import cv2
import numpy as np
import glob
import sys
from sklearn.model_selection import train_test_split

print 'Loading training data........'

e0 = cv2.getTickCount()

#Loading Training Data
image_array = np.zeros((1, 307200))
label_array = np.zeros((1,4), 'float')
training_data = glob.glob('training_data/*.npz')

for single_npz in training_data:
    with np.load(single_npz) as data:
          print data.files
    train_temp = data['train']
    print train_temp
    '''train_labels_temp = data['train_labels']
    print train_temp.shape
    print train_labels_temp.shape
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))
    train = image_array[1:, :]
    train_labels = label_array[1:,:]
    print train.shape
    print train_labels.shape
e00 = cv2.getTickCount()
time0 = (e00 - e0)/cv2.getTickFrequency()
print 'Loading Image duration: ',time0'''
 
