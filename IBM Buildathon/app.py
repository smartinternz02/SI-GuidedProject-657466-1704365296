import numpy as np
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing import  image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from flask import Flask, request, render_template, redirect, url_for
from cloudant.client import Cloudant

model = load_model('Updated-Xception-diabetic-retinopathy.h5')
app = Flask(__name__)

client = Cloudant.iam('a1bddb33-2873-4a20-b891-3bbc7f585d70-bluemix', 'AfJP9ps2RaNhdgeO5bmJdAV0Z_kprgDt4B5vAh_oo_JN' , connect=True)
my_database = client.create_database('my_database')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def home():
    return render_template("index.html")

@app.route('/register' , methods=['GET'])
def register():
    return render_template("register.html")

@app.route('/afterreg', methods=['POST'])
def afterreg():
    x = [x for x in request.form.values()]
    print(x)
    data = {
        '_id' : x[1],
        'name': x[0],
        'psw' : x[2]
    }
    print(data)

    query = {'_id' : {'Seq' : data['_id']}}
    docs  = my_database.get_query_result(query)
    print(docs)
    print(len(docs.all()))

    if (len(docs.all())== 0):
        url = my_database.create_document(data)
        return render_template('register.html' , pred='Registration succesful , please login using your details')
    else:
        return render_template('register.html' , pred='You are already a member, please login using your details')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/afterlogin', methods=['POST'])
def afterlogin():
    user = request.form('_id')
    pssw = request.form('psw')
    print(user,pssw)

    query = {'_id' : {'Seq' : user}}
    docs = my_database.get_query_result(query)
    print(docs)

    print(len(docs.all()))

    if(len(docs.all())==0):
        return render_template('login.html', pred='This username is not found')
    else:
        if(user == docs[0][0]['_id'] and pssw == docs[0][0]['psw']):
            return redirect(url_for('prediction'))
        else:
            print('Invalid user')

@app.route('/logout')
def logout():
    return render_template('logout.html')

@app.route('/result')
def predict():
        return render_template('prediction.html')

@app.route('/result', methods=['GET' , 'POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(299,299))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['No Diabetic Retinopathy', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        result = "DR is:"+str(index[prediction[0]])
        return render_template('prediction.html', prediction=result)

if __name__=='__main__':
    app.run(debug=False)