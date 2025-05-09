from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model pipeline
with open('static/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_value = None
    if request.method == 'POST':
        try:
            year = int(request.form['Year'])
            rainfall = float(request.form['Average_Rainfall'])
            pesticides = float(request.form['Pesticides_Tonnes'])
            temperature = float(request.form['Avg_Temp'])
            area = float(request.form['Area'])
            crop = request.form['Crop']

            # Make DataFrame input for pipeline
            input_df = pd.DataFrame([{
                'Year': year,
                'Rainfall': rainfall,
                'Pesticides': pesticides,
                'Temperature': temperature,
                'Area': area,
                'Crop': crop
            }])

            prediction = model.predict(input_df)
            predicted_value = round(prediction[0], 2)

        except Exception as e:
            predicted_value = f"Error: {str(e)}"

    return render_template('index.html', predicted_value=predicted_value)

if __name__ == '__main__':
    app.run(debug=True)










