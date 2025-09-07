import os
import cv2
import glob
import random
import shutil

from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

# Getting all images to arrays (memory)

train_path="/content/drive/MyDrive/Dataset/Training/"
test_path="/content/drive/MyDrive/Dataset/Testing/"
val_path="/content/drive/MyDrive/Dataset/Validation/"

# 70% is allocated for training
x_train=[]

for folder in os.listdir(train_path):
    sub_path=train_path+"/"+folder

    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        # image colour model BGR to RGB
        img_arr=cv2.resize(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB),(224,224))
        x_train.append(img_arr)

# 15% is allocated for testing
x_test=[]

for folder in os.listdir(test_path):
    sub_path=test_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        # image colour model BGR to RGB
        img_arr=cv2.resize(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB),(224,224))
        x_test.append(img_arr)

# 15% is allocated for validation
x_val=[]

for folder in os.listdir(val_path):
    sub_path=val_path+"/"+folder

    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        # image colour model BGR to RGB
        img_arr=cv2.resize(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB),(224,224))
        x_val.append(img_arr)



#For normalization these are divided by 255
# Normalizing is an effective preprocessing step that can enhance the performance and stability of deep learning models

train_x=np.array(x_train).astype("float32")
test_x=np.array(x_test).astype("float32")
val_x=np.array(x_val).astype("float32")

train_x=train_x/255
test_x=test_x/255
val_x=val_x/255

# Visualize the dataset
print("Shape of training images", train_x.shape) # 70% training
print("Shape of testing images", test_x.shape) # 15% testing
print("Shape of validation images", val_x.shape) # 15% validation

# Image online augmentation applied to training with the help of Keras library
train_datagen = ImageDataGenerator(
    rescale = 1./255, #images are normalized
    zoom_range=[0.8,1.1], #between 20% zoom in and 110% zoom out
    height_shift_range=0.1, # -10% to +10% vertically
    rotation_range=8 #rotate between 0 to 8 degrees
)

test_datagen = ImageDataGenerator(rescale = 1./255) #images are normalized
val_datagen = ImageDataGenerator(rescale = 1./255) #images are normalized



# batch size hyper parameter refers to the number of training examples utilized in one iteration of training
# each iteration of training will process 30 images at once.

batchSize = 30

# with flow_from_directory relevant classes will be automatically identified according to folder names
# 70% of data allocated
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = batchSize,
                                                 class_mode = 'sparse' #returns a 1D array of integer labels for respective classes
                                                 )
# 15% of data allocated
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = batchSize,
                                            class_mode = 'sparse' #returns a 1D array of integer labels for respective classes
                                            )
# 15% of data allocated
val_set = val_datagen.flow_from_directory(val_path,
                                          target_size = (224, 224),
                                          batch_size = batchSize,
                                          class_mode = 'sparse' #returns a 1D array of integer labels for respective classes
                                          )

class_folders = glob('/content/drive/MyDrive/Dataset/Training/*') # number of classes are retrieved


# Set class
train_y=training_set.classes
test_y=test_set.classes
val_y=val_set.classes

# print few training images with classes

for i in range (3):

    plt.imshow(train_x[i])
    plt.show()
    print(train_y[i])

# data summary
import seaborn as sns
sns.countplot(train_y)
plt.title("Labels for classes")

#Prepearing labels
print(train_y.shape)
print(test_y.shape)
print(val_y.shape)

#Print class names
print(training_set.class_indices)

#Model building

# (224, 224, 3) specifies that the model expects input images of size 224x224 pixels with 3 color channels (RGB).
# weights='imagenet' initializes the model with pre-trained weights from the ImageNet dataset, which helps in transfer learning
# include_top=False, it excludes the final fully connected layer of the model, allows to add a custom classification layers
# which will be ideal for this dataset

mobilenet = MobileNet(input_shape=(224,224,3),
                      weights='imagenet',
                      include_top=False)

print("Number of layers " + str(len(mobilenet.layers)))
mobilenet.summary()

#mobilenet fine-tuning

#mobilenet pretrained layers won't be trained
for layer in mobilenet.layers:
    layer.trainable = False

# un-freeze the BatchNorm layers
for layer in mobilenet.layers:
    if "BatchNormalization" in layer.__class__.__name__:
        layer.trainable = True

# Batch Normalization layers often help stabilize the training process.
# By allowing these layers to be trainable can improve performance, as they can adapt to the new data distribution.



#custom final layer

# This converts the output of the MobileNet model which is typically a multi-dimensional tensor into a 1D array.
# This is necessary before connecting to dense layers

x = Flatten()(mobilenet.output)

# The Dropout layer is used here to prevent overfitting by randomly deactivating neurons during training (in this case, 50%)
# This helps improve the model's generalization

x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# A Dense layer with 128 units and ReLU activation introduces learnable parameters, allowing the model to learn complex features.

# This layer outputs the final predictions, with the number of units equal to the number of classes
# The softmax activation function is used for multi-class classification, it's providing probabilities for each class.

prediction = Dense(len(class_folders), activation='softmax')(x)

model = Model(inputs=mobilenet.input, outputs=prediction)

len(class_folders)

print("Number of layers " + str(len(model.layers)))
# view the structure of the model
model.summary()


# compile the model before start training
from tensorflow.keras.optimizers import Adamax

# loss function quantifies how well the model's predictions match the actual target values
# sparse categorical cross-entropy loss function is used since the classes are used in this prototype are mutually exclusive

# Optimization algorithms are used to minimize loss function, a lower value indicates better model performance
# Adamax is an adaptive learning rate optimization algorithm based on Adam.

# The learning rate controls how much to adjust the weights during training
# Accuracy is taken as the metrics during training and evaluation
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adamax(learning_rate=0.001),
    metrics=['accuracy']
)

# In Keras, the get_config() method can be used to retrieve the configuration of an optimizer
model.optimizer.get_config()


#Avoid overfitting the model by stopping early
# This prevents the model learning the training data too well rather than the underlying patterns.
from tensorflow.keras.callbacks import EarlyStopping

early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10)

print(len(x_train)//batchSize, len(x_val)//batchSize)

# train the model

# Steps per epoch specifies how many batches to process in one epoch
# The steps per epoch calculated by dividing the total number of respective samples by the batch size.
# This ensures that all data is utilized effectively across epochs

# Epochs defines the total number of times the training process will run through the entire training dataset
# Although when early stopping is implemented, the actual training may stop before reaching the specified number of epochs.

stepsPerEpoch = len(x_train)//batchSize #(350 x 4) // 30 = 46
validationSteps = len(x_val)//batchSize #(75 x 4) // 30 = 10

# Calculating steps_per_epoch and validation_steps in this way ensures that
# The model efficiently utilizes the dataset by processing it in manageable batches

# the number of epochs are set to 500
history = model.fit(
    training_set,
    steps_per_epoch=stepsPerEpoch,
    validation_data=val_set,
    validation_steps=validationSteps,
    epochs=500,
    callbacks=early_stop
)

# save the structure of the neural network
model_structure = model.to_json()
f = Path("/content/drive/MyDrive/FYP/NewModel/model_structure.json")
f.write_text(model_structure)

# Save trained weights of the neural network
model.save_weights("/content/drive/MyDrive/FYP/NewModel/model_weights.weights.h5")
model.save("/content/drive/MyDrive/FYP/NewModel/model.h5")

# list all data in history
print(history.history.keys())

#model evaluation
model.evaluate(test_x,test_y,batch_size=batchSize)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# Confusion matrix
cm = confusion_matrix(test_y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0','1','2','3','4','5'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("Classification Report:\n")
print(classification_report(test_y, y_pred, target_names=['0','1','2','3','4','5']))

# Accuracy
print("Accuracy:", accuracy_score(test_y, y_pred))

# make predictions
import tensorflow as tf
print(tf.keras.__version__)

from keras.models import model_from_json
from keras.preprocessing import image
from matplotlib import pyplot as plt

class_labels = [
    "Front_View",
    "Non_Front_View",
    "Non_Rear_Bumper",
    "Non_Sedan_Side_View",
    "Rear_Bumper",
    "Sedan_Side_View",
]


# model structure is loaded by the json file
json_file = open('/content/drive/MyDrive/FYP/NewModel/model_structure.json', 'r')
model_structure = json_file.read()
json_file.close()

# with the help of json data keras model object will be re-created
model = model_from_json(model_structure)

# trained weights of the model will be re-loaded
model.load_weights("/content/drive/MyDrive/FYP/NewModel/model_weights.weights.h5")

# image file wiil be loaded to test
# img = image.load_img("/content/drive/MyDrive/FYP/13.jpg", target_size=(224,224,3)) #Shear
# img = image.load_img("/content/drive/MyDrive/FYP/5.jpg", target_size=(224,224,3)) #Flexural
# img = image.load_img("/content/drive/MyDrive/FYP/17.jpg", target_size=(224,224,3)) #Torsional
img = image.load_img("/content/drive/MyDrive/Dataset/Testing/Sedan_Side_View/Toyota car driver door accident_0015.jpg", target_size=(224,224,3)) #No Cracks

# print RGB images
plt.imshow(img)
plt.show()

# image will be converted to numpy array
image_to_test = image.img_to_array(img) / 255

# because image is trained on mini batches to represent the btach size another
# dimension should be added to the single image array
list_of_images = np.expand_dims(image_to_test, axis=0)

# with the use of model the prediction is done
results = model.predict(list_of_images)

print("Possiblility with all classes " + str(results))

# first result will be only checked
single_result = results[0]

print(single_result)

# also get the likehood score for all classes
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]

# most likely class name will be retrieved
class_label = class_labels[most_likely_class_index]

# bolding text
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

print(color.BOLD + "This is a " + color.END + class_label)
print(color.BOLD + "Likehood " + color.END + str(class_likelihood))

# analysis according to the type
if str(class_label) == 'Front_View':
    print(color.BOLD + "Front_View: ")
elif str(class_label) == 'Non_Front_View':
    print(color.BOLD + "Non_Front_View: ")
elif str(class_label) == 'Non_Rear_Bumper':
    print(color.BOLD + "Non_Rear_Bumper: ")
elif str(class_label) == 'Non_Sedan_Side_View':
    print(color.BOLD + "Non_Sedan_Side_View: ")
elif str(class_label) == 'Rear_Bumper':
    print(color.BOLD + "Rear_Bumper: ")
elif str(class_label) == 'Sedan_Side_View':
    print(color.BOLD + "Sedan_Side_View: ")
else:
    exit()