import os
from flask import Flask, render_template, url_for, request
import cv2
import numpy as np
from PIL import Image
from keras import models




app = Flask(__name__)


 

@app.route('/')
@app.route('/home')
def home():
    return render_template("captureImages3.html")



@app.route('/result',methods=['POST', 'GET'])
def result():
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


    return render_template('captureImages3.html', newnew5 = newnew4)
    




if __name__ == "__main__":
    app.run(debug=True)