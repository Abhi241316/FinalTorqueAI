'''

import os
from flask import Flask, render_template, url_for, request




app = Flask(__name__)


 

@app.route('/')
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


    return render_template('captureImages1.html', newnew1 = newnew)
    




if __name__ == "__main__":
    app.run(debug=True)