from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

FEATURE_NAMES = [
    'monsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
    'AgriculturalPractices', 'Encroachments', 'ineffectiveDisasterPreparedness',
    'DrainageSystems', 'Coastal vulnerability', 'Landslides', 'watershades',
    'deteriorating infrastructure', 'population score', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

# Load trained model
with open('flood_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html', feature_names=FEATURE_NAMES)


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form[name]) for name in FEATURE_NAMES]
    except (KeyError, ValueError):
        return render_template(
            'index.html',
            feature_names=FEATURE_NAMES,
            error='Please provide numeric values for all fields.',
        )

    X = np.array(values).reshape(1, -1)
    prediction = model.predict(X)[0]

    risk_label = 'High Risk' if str(prediction).strip().lower() == 'high' else 'Low Risk'

    return render_template(
        'index.html',
        feature_names=FEATURE_NAMES,
        prediction=risk_label,
        input_data=request.form,
    )


if __name__ == '__main__':
    app.run(debug=True)