# Import libraries
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl 
import scipy.io
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.utils import to_categorical
from keras.models import Sequential,  Model,load_model
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, MaxPool1D, GaussianNoise, GlobalMaxPooling1D, Conv1D, GlobalAveragePooling1D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
import math, time


# Function for N X N patches  
def GetPatches_nxn(data, N) :
    
    n = math.floor(N/2)
    n = int(n)
    row, col, ch = data.shape
    num_ele = row * col
    # print("Original ", data.shape)
    padding_top = data[:n, :, :] * 0
    padding_bottom = data[row-n:, :, :] * 0
    data = np.concatenate((padding_top, data, padding_bottom), axis= 0)

    row, col, ch = data.shape
    padding_left = data[:, :n, :] * 0
    padding_right = data[:, col-n:, :] * 0
    data = np.concatenate((padding_left, data, padding_right), axis= 1)
    # print("Final ", data.shape)
    patches = np.zeros((num_ele, N, N, 2)) + np.nan
    num_nans = np.count_nonzero(np.isnan(patches))
    # print(num_nans)
    row, col, ch = data.shape
    count = 0
    for i in range(n, row-n) :
        for j in range(n, col-n) :
            
            # print(i,j)
            # print(i-n, i+n+1)
            # print(j-n, j+n+1)
            # print("i ", i, " j ", j, "row ", row, " col ", col, " top ", i-n, " bottom ", i+n+1, " left ", j-n, " right ", j+n+1)

            assert i-n >= 0
            assert j-n >= 0
            assert i+n+1 <= row
            assert j+n+1 <= col
            patch = data[i-n:i+n+1, j-n:j+n+1, :]
            assert patch.shape == (N,N,2)
            patches[count] = patch
            count += 1

    patches = np.array(patches)

    return patches



# Get path to your current directory
basedir = os.getcwd()

# Path to your dataset
filename = basedir + "/muufl_gulfport_campus_1_hsi_220_label.mat"

# Open .mat file with scipy
mat = scipy.io.loadmat(filename)

hsi = ((mat['hsi'])[0])[0]

# RGB Image
rgbIm = hsi[-1]

# Ground truth
truth = ((hsi[-2])[0])[-1]
truth = truth[-1]

# LiDAR
lidar = ((((hsi[-4])[0])[0])[0])[0]

# x, y, z. z contains Height and Intensity
x, y, z, info = lidar[0], lidar[1], lidar[2], lidar[3]

patches = GetPatches_nxn(z, 11)
print(patches.shape)

# Ground truth contains label -1, 1, 2, ..., 11. Label '-1' is unlabelled data. So, we subtract 1 from ground truth.
# Now, the ground truth becomes -2, 0, 1, ..., 10. And we take ground truth >=0.
truth = truth.flatten()
truth = truth - 1
indx, = np.where(truth >= 0)
patches = patches[indx]
truth = truth[indx]


# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(patches, truth, test_size= 0.3, random_state = int(time.time()), shuffle = True)

np.savez_compressed(basedir +"/train_test_split_lidar11_iter1.npz", X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)

file = np.load(basedir+"/train_test_split_lidar11_iter1.npz")
X_train = file['X_train']
X_test = file['X_test']
y_train = file['y_train']
y_test = file['y_test']

# Normalize the data
ch1 = X_train[:, :, :, 0]
pmin = np.amin(ch1)
pmax = np.amax(ch1)
ch1 = (ch1-pmin) / (pmax- pmin) 


ch2 = X_train[:, :, :, 1]
pmin1 = np.amin(ch2)
pmax1 = np.amax(ch2)
ch2 = (ch2-pmin1) / (pmax1- pmin1) 
X_train[:,:,:,0] = ch1
X_train[:,:,:,1] = ch2

ch3 = X_test[:, :, :, 0]
ch3 = (ch3-pmin) / (pmax- pmin) 

ch4 = X_test[:, :, :, 1]
ch4 = (ch4-pmin1) / (pmax1- pmin1) 

X_test[:,:,:,0] = ch3
X_test[:,:,:,1] = ch4

print(np.unique(y_train, return_counts= True))

# Since the data is imbalanced we filter out the labels that have total indices less than threshold and repeat them till threshold.
u, f = np.unique(y_train, return_counts= True)
useful_indices = []
thresh = 4000
for i in range(len(u)) :

    indx, = np.where(y_train == u[i])
    indx = indx.tolist()
    if f[i] < thresh :
        while len(indx) < thresh :
            indx = indx + indx
        indx = indx[:thresh]
        useful_indices += indx
    else :
        useful_indices += indx

X_train = X_train[useful_indices]
y_train = y_train[useful_indices]
print(np.unique(y_train, return_counts= True))

# One hot encoding of labels
y_train = to_categorical(y_train,num_classes = 11, dtype ="int32")

y_test = to_categorical(y_test,num_classes = 11, dtype ="int32")

# print(X_train.shape)

input = X_train[0]
print(input.shape)

def get_model():

    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape = input.shape, activation = 'tanh', padding = 'same'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation = 'tanh', padding = 'same'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation = 'tanh', padding = 'same'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(32, activation= 'tanh'))
    model.add(BatchNormalization())
    model.add(Dense(11, activation= 'softmax'))


    model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy', metrics=['categorical_accuracy'])

    model.summary()

    return model



model = get_model()
model.fit(X_train, y_train, epochs= 20, batch_size= 256, verbose= 1)
model.save(basedir + "/lidar_model11_iter1.h5")

model = load_model(basedir+ "/lidar_model11_iter1.h5")

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis= -1)
y_test = np.argmax(y_test, axis= -1)

correct = len(y_pred) - np.count_nonzero(y_pred - y_test)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100
   
print("Accuracy: ", acc)

class_names = ['Trees', 'Mostly grass', 'Mixed ground', 'Dirt and sand', 'road', 'water', 'building shadow', 'building', 'sidewalk', 'yellow curb', 'cloth panels']

cm = confusion_matrix(y_test, y_pred, normalize= 'true')
cm = np.round(cm, 3)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot()
plt.show()
