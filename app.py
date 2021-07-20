# import libraires
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from flask import Flask , render_template , request
import gunicorn

#load model
model = pickle.load(open('Calories_model_RFR.pkl', "rb"))

#load scaler
scalerfile = 'scaler.save'
scaler = pickle.load(open(scalerfile, 'rb'))


#flask consructor
app = Flask(__name__)

@app.route('/')

@app.route('/main_template',methods=["GET"])
def main_template():

    #render form
    return render_template('Index.html')

#get form data
@app.route('/predict',methods=['GET','POST'])
def predict():

    #checking request type
    str_req_type = request.method

    #convert string value into numeric value
    if request.method == str(str_req_type):

        if request.args.get('gender') == 'Male':
            gender = 1

        else:
            gender = 0

        age = request.args.get('age')

        duration = request.args.get('duration')

        heart_Rate = request.args.get('heart_rate')

        temp = request.args.get('temp')

        height = request.args.get('height')

        weight = request.args.get('weight')

        #store form values into set
        values = [float(gender), float(age), float(height), float(weight), float(duration), float(heart_Rate), float(temp)]

        #turn into array & reshape array for prediction
        input_array = np.asarray(values)
        input_array_reshape = input_array.reshape(1, -1)

        #sclae the inputed reshaped data
        scaled_set = scaler.transform(input_array_reshape)

        # predict with inputed values
        predicted = model.predict(scaled_set)

        #display predicted valuesin result.html file
        return  render_template('result.html', predicted_value=predicted[0])

    else:

        return render_template('Index.html')


if __name__ == '__main__':
    app.run(debug=True)
