import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Main parameters
dataPath="trafficSigns" 
labelFile='labels.csv' 
batchSizeVal=50 
stepsPerEpochVal=446 
epochs_val=10
imageDimesions=(32,32,3)
testRatio=0.2
validationRatio=0.2 

#Importing the images
count=0
images=[]
classNo=[]
classList=os.listdir(dataPath)
noOfClasses=len(classList)-1

print("Total Classes Detected:",noOfClasses)
print("Importing Classes.....")
for i in range(0,noOfClasses):
    imgList=os.listdir(dataPath+"/"+str(count))
    for j in imgList:
        currImg=cv2.imread(dataPath+"/"+str(count)+"/"+j)
        images.append(currImg)
        classNo.append(count)
    print(count,end=" ")
    count+=1
print(" ")

images=np.array(images)
classNo=np.array(classNo)

#Splitting the data
xTrain,xTest,yTrain,yTest=train_test_split(images,classNo,test_size=testRatio)
xTrain,xValidation,yTrain,yValidatiob=train_test_split(xTrain,yTrain,test_size=validationRatio)

#xTrain array of imgs to train
#yTrain corresponding classs ids

#Labels and images test
print("Data Shapes")
print("Train",end="")
print(xTrain.shape,yTrain.shape)
print("Validation",end="")
print(xValidation.shape,yValidatiob.shape)
print("Test",end="")
print(xTest.shape,yTest.shape)
assert(xTrain.shape[0]==yTrain.shape[0]),"The number of images in not equal to the number of lables in training set"
assert(xValidation.shape[0]==yValidatiob.shape[0]),"The number of images in not equal to the number of lables in validation set"
assert(xTest.shape[0]==yTest.shape[0]),"The number of images in not equal to the number of lables in test set"
assert(xTrain.shape[1:]==(imageDimesions)),"The dimesions of the Training images are wrong"
assert(xValidation.shape[1:]==(imageDimesions)),"The dimesionas of the Validation images are wrong"
assert(xTest.shape[1:]==(imageDimesions)),"The dimesionas of the Test images are wrong"

#Read the CSV file
data=pd.read_csv(labelFile)
print("data shape ",data.shape,type(data))

#Preprocessing an image
def grayscale(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img=cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img=grayscale(img) 
    img=equalize(img)  # Standarize the lighting in the img
    img=img/255 #Change values into percents
    return img

#Preprocessing all data 
xTrain=np.array(list(map(preprocessing,xTrain)))
xValidation=np.array(list(map(preprocessing,xValidation)))
xTest=np.array(list(map(preprocessing,xTest)))

#Adding one dimension 
xTrain=xTrain.reshape(xTrain.shape[0],xTrain.shape[1],xTrain.shape[2],1)
xValidation=xValidation.reshape(xValidation.shape[0],xValidation.shape[1],xValidation.shape[2],1)
xTest=xTest.reshape(xTest.shape[0],xTest.shape[1],xTest.shape[2],1)

dataGen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
dataGen.fit(xTrain)
batches=dataGen.flow(xTrain,yTrain,batch_size=20)
xBatch,yBatch=next(batches)

#Creating a matrix
yTrain=to_categorical(yTrain,noOfClasses)
yValidatiob=to_categorical(yValidatiob,noOfClasses)
yTest=to_categorical(yTest,noOfClasses)


#Defining model
def myModel():
    noOfFilters=60
    sizeOfFitler=(5,5)
    sizeOfFitler2=(3,3)
    sizeOfPool=(2,2)
    noOfNodes=500
    model=Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFitler,input_shape=(imageDimesions[0],imageDimesions[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters,sizeOfFitler,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))

    model.add((Conv2D(noOfFilters//2,sizeOfFitler2,activation='relu')))
    model.add((Conv2D(noOfFilters//2,sizeOfFitler2,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))

    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

#Training the model baby
model=myModel()
print(model.summary())

history=model.fit_generator(dataGen.flow(xTrain,yTrain,batch_size=batchSizeVal),steps_per_epoch=stepsPerEpochVal,epochs=epochs_val,validation_data=(xValidation,yValidatiob),shuffle=1)

score=model.evaluate(xTest,yTest,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])

#Saving the model
model.save('model.h5')