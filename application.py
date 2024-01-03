from flask import Flask,render_template,request
import numpy as np
import pickle
# create a flask object
application=Flask(__name__)
'''
@application.route('/')
def hello():
    """test function"""
    return "Welcome to the Flask"

@application.route('/shoheb',methods=['GET'])
def check():
    """new function"""
    return "Codegnan is in KITS College"
'''
#First let's read the pickle file
with open('House_Price.pkl','rb') as f:
    model=pickle.load(f)
@application.route('/',methods = ['GET'])
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
    Rooms=int(request.form['bedrooms'])# python left html right
    Bathrooms=int(request.form['bathrooms'])
    Place=int(request.form['location'])
    Area=int(request.form['area'])
    Status=int(request.form['status'])
    Facing=int(request.form['facing'])
    P_Type=int(request.form['type'])
#Now take the above form data and convert to array
    input_data=np.array([[Place,Area,Status,Rooms,Bathrooms,Facing,P_Type]])
#by taking above data we will predict the House_price
    prediction=model.predict(input_data)[0]
#now we will pass the above predicted data to template
    return render_template("index.html",prediction=prediction)
if __name__=="__main__":
    application.run()
