from flask import Flask, request, render_template
import numpy as np

from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app= application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        try:
            data = CustomData(
                session_time=float(request.form.get("session_time")),
                clicks=float(request.form.get("clicks")),
                pages_visited=float(request.form.get("pages_visited")),
                last_login_days=float(request.form.get("last_login_days")),
                purchase_count=float(request.form.get("purchase_count"))
            )

            final_data = data.get_data_as_dataframe()

            pipeline = PredictPipeline()
            pred, prob = pipeline.predict(final_data)

            result = "High Risk" if pred[0] == 1 else "Low Risk"

            return render_template(
                'home.html',
                results=result,
                probability=round(float(prob[0]), 2)
            )

        except Exception as e:
            return str(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)