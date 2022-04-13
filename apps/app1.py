import os
from flask import Flask, render_template, url_for, request
import sys
import cv2
from PIL import Image



app = Flask(__name__)

 

@app.route('/')
@app.route('/home')
def home():
    return render_template("captureImages.html")



@app.route('/result',methods=['POST', 'GET'])
def result():
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

    return render_template('captureImages.html', newnew3 = newnew2)






if __name__ == "__main__":
    app.run(debug=True)