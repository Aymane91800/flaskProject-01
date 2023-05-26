import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():  # put application's code here
    if request.method == 'POST':
        try:
            volume = int(request.form['volume'])
            weight = int(request.form['weight'])

            pred_arr = np.array([volume, weight])
            preds = pred_arr.reshape(1, -1)
            preds = preds.astype(int)
            model = open('lin_reg.pkl', 'rb')
            lr_model = joblib.load(model)
            model_prediction = lr_model.predict(preds)
            model_prediction = round(float(model_prediction), 2)
        except ValueError:
            return "Please Enter Valid Values"

    return render_template('predict.html', prediction=model_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
