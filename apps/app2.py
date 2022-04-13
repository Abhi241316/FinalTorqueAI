import os
from flask import Flask, render_template, url_for, request
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




app = Flask(__name__)


 

@app.route('/')
@app.route('/home')
def home():
    return render_template("captureImages2.html")



@app.route('/result',methods=['POST', 'GET'])
def result():
    output = request.form.to_dict()
    newnew2= output["name"]
    newnew3= output["name"]
    

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
    newnew3= output["name"]
    # fit the model
    r = model.fit(
      training_set,
      validation_data=test_set,
      epochs=int(newnew3),
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


    return render_template('captureImages2.html', newnew3 = newnew2)
    




if __name__ == "__main__":
    app.run(debug=True)