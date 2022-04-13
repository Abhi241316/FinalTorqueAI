# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_migrate import Migrate
from sys import exit
from decouple import config

from apps.config import config_dict
from apps import create_app, db

import os
from flask import Flask, render_template, url_for, request
import cv2
from PIL import Image

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras import models


# WARNING: Don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True, cast=bool)

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:

    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)
Migrate(app, db)

if DEBUG:
    app.logger.info('DEBUG       = ' + str(DEBUG))
    app.logger.info('Environment = ' + get_config_mode)
    app.logger.info('DBMS        = ' + app_config.SQLALCHEMY_DATABASE_URI)


@app.route('/home')
def home():
    return render_template("captureImages1.html")



@app.route('/result',methods=['POST', 'GET'])
def result():
    output = request.form.to_dict()
    newnew = output["name"]
    newnew1= output["name"]
    if newnew != "quit":
        os.mkdir(newnew1)

    test1="Folder Created !"
    return render_template('captureImages1.html', test1 = test1)
    


@app.route('/home1')
def home1():
    return render_template("captureImages.html")


@app.route('/result1',methods=['POST', 'GET'])
def result1():
    output = request.form.to_dict()

    
    newnew2= output["name1"]
    newnew3= output["name1"]
    newnewnew= output["name0"]
    newnewnew1= output["name"]
    directory = newnewnew1
    imagecount = int(newnewnew)

    os.makedirs(directory, exist_ok=True)

    newnew3=output["name1"]
    video = cv2.VideoCapture(newnew3)

    filename = len(os.listdir(directory))
    count = 0

    while True and count < imagecount:
        filename += 1
        count += 1
        _, frame = video.read()
        im = Image.fromarray(frame, 'RGB')
        im = im.resize((224, 224))
        im.save(os.path.join(directory, str(filename) + ".jpg"), "JPEG")

        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

    test2="Data Captured !"

    return render_template('captureImages.html', test2=test2)



@app.route('/home2')
def home2():
    return render_template("captureImages2.html")



@app.route('/result2',methods=['POST', 'GET'])
def result2():
    output = request.form.to_dict()
    newnew22= output["name"]
    newnew33= output["name"]
    

    # re-size all the images to this
    IMAGE_SIZE = [224, 224]

    #datasetstemp=input("Enter the directory for training (database path)  :  ")

    train_path = 'Datasets/Train'
    valid_path = 'Datasets/Test'

    # add preprocessing layer to the front of VGG
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in vgg.layers:
      layer.trainable = False
      

      
      # useful for getting number of classes
    folders = glob('Datasets/Train/*')
      

    # our layers - you can add more if you want
    x = Flatten()(vgg.output)
    # x = Dense(1000, activation='relu')(x)
    prediction = Dense(len(folders), activation='softmax')(x)

    # create a model object
    model = Model(inputs=vgg.input, outputs=prediction)

    # view the structure of the model
    model.summary()

    # tell the model what cost and optimization method to use
    model.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
    )


    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('Datasets/Train',
                                                     target_size = (224, 224),
                                                     batch_size = 32,
                                                     class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory('Datasets/Test',
                                                target_size = (224, 224),
                                                batch_size = 32,
                                                class_mode = 'categorical')

    '''r=model.fit_generator(training_set,
                             samples_per_epoch = 8000,
                             nb_epoch = 5,
                             validation_data = test_set,
                             nb_val_samples = 2000)'''
    newnew33= output["name"]
    # fit the model
    r = model.fit(
      training_set,
      validation_data=test_set,
      epochs=int(newnew33),
      steps_per_epoch=len(training_set),
      validation_steps=len(test_set)
    )


    import tensorflow as tf

    from keras.models import load_model

    temp11=output["name1"]
    temp22=output["name2"]

    model_name=temp11
    model.save(model_name)

    # Saving the model for Future Inferences
    tempjson=temp22
    model_json = model.to_json()
    with open(tempjson, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5

    test3="Training Completed! Model saved"

    return render_template('captureImages2.html', test3=test3)


@app.route('/home3')
def home3():
    return render_template("captureImages3.html")



@app.route('/result3',methods=['POST', 'GET'])
def result3():
    output = request.form.to_dict()
    newnew1=output["name"]

    model_name=newnew1

    #Load the saved model
    model = models.load_model(model_name)

    newnew2=output["name1"]

    newnew=newnew2
    
    video = cv2.VideoCapture(newnew)
    
    newnew3=output["name2"]

    lbl1=newnew3

    newnew4=output["name3"]
    newnew5=output["name3"]
    
    lbl2=newnew5
    while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
        
        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((224,224))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        #img_array = np.expand_dims(img_array, axis=0)
        
        #Calling the predict method on model to predict 'me' on the image
        prediction = int(model.predict(np.expand_dims(img_array,axis=0))[0][0])
        
        #if prediction is 0, which mean I am missing on the image, then show the frame in gray color.
        

        if prediction==0:
                    print(lbl1)
        if prediction==1:
            print(lbl2)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

    test4="Detection Done!"

    return render_template('captureImages3.html', test4=test4)
    

@app.route("/namehai")
def namehais():
    return render_template("captureImages.html")


@app.route("/namehai1")
def namehais1():
    return render_template("captureImages2.html")


@app.route("/namehai2")
def namehais2():
    return render_template("captureImages3.html")

@app.route("/namehai3")
def namehais3():
    return render_template("captureImages1.html")


if __name__ == "__main__":
    app.run()
