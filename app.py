from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import requests
from datetime import datetime
from roadmap_engine import IMPROVEMENTS, apply_improvements, calculate_risk, suggest_roadmap

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


# Regional baseline profiles for scenario simulation
REGIONAL_PROFILES = {
    'Bangladesh': {'TopographyDrainage': 12, 'RiverManagement': 10, 'Deforestation': 9, 'Urbanization': 10,
                  'DamsQuality': 7, 'Siltation': 11, 'Encroachments': 10, 'ineffectiveDisasterPreparedness': 9,
                  'DrainageSystems': 8, 'Coastal vulnerability': 11, 'Landslides': 6, 'watershades': 10,
                  'deteriorating infrastructure': 10, 'population score': 11, 'WetlandLoss': 9, 'AgriculturalPractices': 8,
                  'InadequatePlanning': 9, 'PoliticalFactors': 8},
    'India': {'TopographyDrainage': 10, 'RiverManagement': 8, 'Deforestation': 8, 'Urbanization': 9,
             'DamsQuality': 6, 'Siltation': 9, 'Encroachments': 8, 'ineffectiveDisasterPreparedness': 8,
             'DrainageSystems': 7, 'Coastal vulnerability': 9, 'Landslides': 8, 'watershades': 9,
             'deteriorating infrastructure': 8, 'population score': 10, 'WetlandLoss': 8, 'AgriculturalPractices': 7,
             'InadequatePlanning': 8, 'PoliticalFactors': 6},
    'Thailand': {'TopographyDrainage': 10, 'RiverManagement': 7, 'Deforestation': 8, 'Urbanization': 8,
                'DamsQuality': 6, 'Siltation': 10, 'Encroachments': 7, 'ineffectiveDisasterPreparedness': 7,
                'DrainageSystems': 6, 'Coastal vulnerability': 8, 'Landslides': 7, 'watershades': 9,
                'deteriorating infrastructure': 7, 'population score': 8, 'WetlandLoss': 7, 'AgriculturalPractices': 8,
                'InadequatePlanning': 7, 'PoliticalFactors': 6},
    'Indonesia': {'TopographyDrainage': 11, 'RiverManagement': 6, 'Deforestation': 10, 'Urbanization': 8,
                 'DamsQuality': 5, 'Siltation': 10, 'Encroachments': 8, 'ineffectiveDisasterPreparedness': 8,
                 'DrainageSystems': 6, 'Coastal vulnerability': 10, 'Landslides': 10, 'watershades': 10,
                 'deteriorating infrastructure': 7, 'population score': 9, 'WetlandLoss': 9, 'AgriculturalPractices': 8,
                 'InadequatePlanning': 7, 'PoliticalFactors': 7},
    'Nigeria': {'TopographyDrainage': 9, 'RiverManagement': 7, 'Deforestation': 8, 'Urbanization': 9,
               'DamsQuality': 6, 'Siltation': 8, 'Encroachments': 9, 'ineffectiveDisasterPreparedness': 9,
               'DrainageSystems': 6, 'Coastal vulnerability': 8, 'Landslides': 5, 'watershades': 8,
               'deteriorating infrastructure': 8, 'population score': 10, 'WetlandLoss': 7, 'AgriculturalPractices': 7,
               'InadequatePlanning': 9, 'PoliticalFactors': 8},
    'United States': {'TopographyDrainage': 6, 'RiverManagement': 5, 'Deforestation': 3, 'Urbanization': 8,
                     'DamsQuality': 3, 'Siltation': 4, 'Encroachments': 5, 'ineffectiveDisasterPreparedness': 3,
                     'DrainageSystems': 4, 'Coastal vulnerability': 6, 'Landslides': 5, 'watershades': 5,
                     'deteriorating infrastructure': 4, 'population score': 7, 'WetlandLoss': 3, 'AgriculturalPractices': 4,
                     'InadequatePlanning': 3, 'PoliticalFactors': 2},
    'United Kingdom': {'TopographyDrainage': 5, 'RiverManagement': 4, 'Deforestation': 2, 'Urbanization': 7,
                      'DamsQuality': 2, 'Siltation': 3, 'Encroachments': 3, 'ineffectiveDisasterPreparedness': 2,
                      'DrainageSystems': 3, 'Coastal vulnerability': 5, 'Landslides': 3, 'watershades': 4,
                      'deteriorating infrastructure': 3, 'population score': 6, 'WetlandLoss': 2, 'AgriculturalPractices': 3,
                      'InadequatePlanning': 2, 'PoliticalFactors': 1},
    'Japan': {'TopographyDrainage': 8, 'RiverManagement': 4, 'Deforestation': 2, 'Urbanization': 8,
             'DamsQuality': 2, 'Siltation': 4, 'Encroachments': 3, 'ineffectiveDisasterPreparedness': 2,
             'DrainageSystems': 2, 'Coastal vulnerability': 8, 'Landslides': 7, 'watershades': 6,
             'deteriorating infrastructure': 2, 'population score': 6, 'WetlandLoss': 2, 'AgriculturalPractices': 3,
             'InadequatePlanning': 2, 'PoliticalFactors': 1},
    'Australia': {'TopographyDrainage': 7, 'RiverManagement': 5, 'Deforestation': 4, 'Urbanization': 7,
                 'DamsQuality': 3, 'Siltation': 5, 'Encroachments': 4, 'ineffectiveDisasterPreparedness': 3,
                 'DrainageSystems': 4, 'Coastal vulnerability': 7, 'Landslides': 4, 'watershades': 6,
                 'deteriorating infrastructure': 3, 'population score': 6, 'WetlandLoss': 4, 'AgriculturalPractices': 5,
                 'InadequatePlanning': 3, 'PoliticalFactors': 2},
}


def rainfall_to_features(rainfall_mm, region='Bangladesh'):
    """
    Convert rainfall amount (mm) to flood risk feature values.
    
    Args:
        rainfall_mm: Rainfall amount in millimeters (0-300)
        region: Region name for baseline adjustments
    
    Returns:
        Dictionary of feature values (0-14 scale)
    """
    # Get baseline profile for region
    base = REGIONAL_PROFILES.get(region, REGIONAL_PROFILES['India'])
    
    # Derived features based on rainfall
    # Monsoon intensity: maps 0-300mm to 0-14 scale
    monsoon_intensity = min(max(0, (rainfall_mm / 50) * 14), 14)
    
    # Climate change proxy: increase with extreme rainfall
    climate_change = min(max(0, ((rainfall_mm - 100) / 100) * 8 + 5), 14) if rainfall_mm > 100 else max(0, (rainfall_mm / 100) * 5)
    
    # River management stress increases with rainfall
    river_mgmt_adjustment = min((rainfall_mm / 150) * 4, 4)  # Up to +4
    
    # Siltation increases with rainfall (erosion)
    siltation_adjustment = min((rainfall_mm / 200) * 3, 3)  # Up to +3
    
    # Coastal vulnerability increases significantly for extreme rainfall
    coastal_adjustment = min((rainfall_mm / 250) * 4, 4) if rainfall_mm > 100 else 0
    
    # Drainage system stress
    drainage_adjustment = min((rainfall_mm / 180) * 3, 3)
    
    # Build feature dictionary
    features = {
        'monsoonIntensity': round(monsoon_intensity, 1),
        'ClimateChange': round(climate_change, 1),
        'TopographyDrainage': round(base['TopographyDrainage'] + 0, 1),
        'RiverManagement': round(min(base['RiverManagement'] + river_mgmt_adjustment, 14), 1),
        'Deforestation': round(base['Deforestation'], 1),
        'Urbanization': round(base['Urbanization'], 1),
        'DamsQuality': round(base['DamsQuality'], 1),
        'Siltation': round(min(base['Siltation'] + siltation_adjustment, 14), 1),
        'AgriculturalPractices': round(base['AgriculturalPractices'], 1),
        'Encroachments': round(base['Encroachments'], 1),
        'ineffectiveDisasterPreparedness': round(base['ineffectiveDisasterPreparedness'], 1),
        'DrainageSystems': round(min(base['DrainageSystems'] + drainage_adjustment, 14), 1),
        'Coastal vulnerability': round(min(base['Coastal vulnerability'] + coastal_adjustment, 14), 1),
        'Landslides': round(base['Landslides'], 1),
        'watershades': round(base['watershades'], 1),
        'deteriorating infrastructure': round(base['deteriorating infrastructure'], 1),
        'population score': round(base['population score'], 1),
        'WetlandLoss': round(base['WetlandLoss'], 1),
        'InadequatePlanning': round(base['InadequatePlanning'], 1),
        'PoliticalFactors': round(base['PoliticalFactors'], 1),
    }
    
    return features


@app.route('/')
def index():
    return render_template('index.html', feature_names=FEATURE_NAMES)


@app.route('/simulator')
def simulator():
    return render_template('simulator.html')


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/roadmap')
def roadmap():
    return render_template('roadmap.html', regions=list(REGIONAL_PROFILES.keys()), improvements=IMPROVEMENTS)

@app.route('/generate_roadmap', methods=['POST'])
def generate_roadmap():
    try:
        data = request.get_json()
        rainfall_mm = float(data.get('rainfall_mm', 0))
        region = data.get('region', 'Bangladesh')
        selected_ids = data.get('selected_improvements', [])
        if not isinstance(selected_ids, list):
            selected_ids = []

        budget = data.get('budget', None)
        if budget is not None and budget != '':
            try:
                budget = float(budget)
            except ValueError:
                budget = None
        else:
            budget = None

        if rainfall_mm < 0 or rainfall_mm > 300:
            return jsonify({'success': False, 'error': 'Rainfall must be between 0 and 300 mm'}), 400

        baseline_features = rainfall_to_features(rainfall_mm, region)
        baseline_risk = calculate_risk(model, baseline_features, FEATURE_NAMES)

        improved_features = apply_improvements(baseline_features, selected_ids) if selected_ids else baseline_features
        improved_risk = calculate_risk(model, improved_features, FEATURE_NAMES)

        selected_improvements = []
        selected_cost_usd = 0
        for imp_id in selected_ids:
            imp = next((item for item in IMPROVEMENTS if item['id'] == imp_id), None)
            if imp:
                selected_cost_usd += imp['cost_usd']
                selected_improvements.append({
                    'id': imp['id'],
                    'name': imp['name'],
                    'category': imp['category'],
                    'cost_usd': imp['cost_usd'],
                    'cost_display': f"${imp['cost_usd']:,}",
                    'timeline_years': imp['timeline_years'],
                    'impact': imp['impact']
                })

        roadmap_data = suggest_roadmap(model, baseline_features, FEATURE_NAMES, budget)

        return jsonify({
            'success': True,
            'region': region,
            'rainfall_mm': rainfall_mm,
            'baseline_risk': baseline_risk,
            'improved_risk': improved_risk,
            'selected_improvements': selected_improvements,
            'selected_cost_usd': selected_cost_usd,
            'recommended_improvements': roadmap_data['recommendations'],
            'budget_usd': roadmap_data['budget_usd'],
            'budget_remaining_usd': roadmap_data['budget_remaining_usd'],
            'total_cost_usd': roadmap_data['total_cost_usd']
        })
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Roadmap generation error: {str(e)}'}), 500

@app.route('/get_weather', methods=['POST'])
def get_weather():
    """
    Fetch weather data using Open-Meteo API and auto-suggest regional risk factors.
    Expects JSON with 'city' field.
    """
    try:
        data = request.get_json()
        city = data.get('city', '').strip()
        
        if not city:
            return jsonify({'error': 'City name is required'}), 400
        
        # Step 1: Get city coordinates using geocoding
        geo_url = f'https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json'
        geo_response = requests.get(geo_url, timeout=5)
        geo_data = geo_response.json()
        
        if not geo_data.get('results'):
            return jsonify({'error': f'City "{city}" not found. Please check the spelling.'}), 404
        
        location = geo_data['results'][0]
        latitude = location['latitude']
        longitude = location['longitude']
        city_name = location.get('name', city)
        country = location.get('country', '')
        
        # Step 2: Get weather data using Open-Meteo
        weather_url = f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,weather_code,cloud_cover&timezone=auto'
        weather_response = requests.get(weather_url, timeout=5)
        weather_data = weather_response.json()
        
        current = weather_data.get('current', {})
        temperature = current.get('temperature_2m', 'N/A')
        humidity = current.get('relative_humidity_2m', 0)
        clouds = current.get('cloud_cover', 0)
        weather_code = current.get('weather_code', 0)
        
        # Estimate rainfall from humidity and cloud cover
        rainfall_mm = max(0, (humidity / 100) * (clouds / 100) * 10)
        monsoon_intensity = min(round((rainfall_mm / 50) * 14, 1), 14)
        climate_change_score = min(round((clouds / 100) * 10 + (humidity / 100) * 4, 1), 14)
        
        # Weather descriptions
        weather_descriptions = {
            0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
            45: 'Foggy', 48: 'Depositing rime fog', 51: 'Light drizzle', 53: 'Moderate drizzle',
            55: 'Dense drizzle', 61: 'Light rain', 63: 'Moderate rain', 65: 'Heavy rain',
            71: 'Light snow', 77: 'Snow grains', 80: 'Light rain showers', 81: 'Moderate rain showers',
            82: 'Violent rain showers', 85: 'Light snow showers', 86: 'Heavy snow showers', 95: 'Thunderstorm'
        }
        description = weather_descriptions.get(weather_code, 'Unknown')
        
        # Step 3: Generate region-based risk profiles
        regional_profiles = {
            # South Asia (High Flood Risk - Monsoon + Rivers + Development)
            'Bangladesh': {'TopographyDrainage': 12, 'RiverManagement': 10, 'Deforestation': 9, 'Urbanization': 10,
                          'DamsQuality': 7, 'Siltation': 11, 'Encroachments': 10, 'ineffectiveDisasterPreparedness': 9,
                          'DrainageSystems': 8, 'Coastal vulnerability': 11, 'Landslides': 6, 'watershades': 10,
                          'deteriorating infrastructure': 10, 'population score': 11, 'WetlandLoss': 9, 'AgriculturalPractices': 8,
                          'InadequatePlanning': 9, 'PoliticalFactors': 8},
            'India': {'TopographyDrainage': 10, 'RiverManagement': 8, 'Deforestation': 8, 'Urbanization': 9,
                     'DamsQuality': 6, 'Siltation': 9, 'Encroachments': 8, 'ineffectiveDisasterPreparedness': 8,
                     'DrainageSystems': 7, 'Coastal vulnerability': 9, 'Landslides': 8, 'watershades': 9,
                     'deteriorating infrastructure': 8, 'population score': 10, 'WetlandLoss': 8, 'AgriculturalPractices': 7,
                     'InadequatePlanning': 8, 'PoliticalFactors': 6},
            
            # Southeast Asia (High Risk - Tropical + Deltas)
            'Thailand': {'TopographyDrainage': 10, 'RiverManagement': 7, 'Deforestation': 8, 'Urbanization': 8,
                        'DamsQuality': 6, 'Siltation': 10, 'Encroachments': 7, 'ineffectiveDisasterPreparedness': 7,
                        'DrainageSystems': 6, 'Coastal vulnerability': 8, 'Landslides': 7, 'watershades': 9,
                        'deteriorating infrastructure': 7, 'population score': 8, 'WetlandLoss': 7, 'AgriculturalPractices': 8,
                        'InadequatePlanning': 7, 'PoliticalFactors': 6},
            'Indonesia': {'TopographyDrainage': 11, 'RiverManagement': 6, 'Deforestation': 10, 'Urbanization': 8,
                         'DamsQuality': 5, 'Siltation': 10, 'Encroachments': 8, 'ineffectiveDisasterPreparedness': 8,
                         'DrainageSystems': 6, 'Coastal vulnerability': 10, 'Landslides': 10, 'watershades': 10,
                         'deteriorating infrastructure': 7, 'population score': 9, 'WetlandLoss': 9, 'AgriculturalPractices': 8,
                         'InadequatePlanning': 7, 'PoliticalFactors': 7},
            
            # Africa (Variable - Some regions high risk)
            'Nigeria': {'TopographyDrainage': 9, 'RiverManagement': 7, 'Deforestation': 8, 'Urbanization': 9,
                       'DamsQuality': 6, 'Siltation': 8, 'Encroachments': 9, 'ineffectiveDisasterPreparedness': 9,
                       'DrainageSystems': 6, 'Coastal vulnerability': 8, 'Landslides': 5, 'watershades': 8,
                       'deteriorating infrastructure': 8, 'population score': 10, 'WetlandLoss': 7, 'AgriculturalPractices': 7,
                       'InadequatePlanning': 9, 'PoliticalFactors': 8},
            
            # Developed Countries (Lower Risk - Better Infrastructure)
            'United States': {'TopographyDrainage': 6, 'RiverManagement': 5, 'Deforestation': 3, 'Urbanization': 8,
                             'DamsQuality': 3, 'Siltation': 4, 'Encroachments': 5, 'ineffectiveDisasterPreparedness': 3,
                             'DrainageSystems': 4, 'Coastal vulnerability': 6, 'Landslides': 5, 'watershades': 5,
                             'deteriorating infrastructure': 4, 'population score': 7, 'WetlandLoss': 3, 'AgriculturalPractices': 4,
                             'InadequatePlanning': 3, 'PoliticalFactors': 2},
            'United Kingdom': {'TopographyDrainage': 5, 'RiverManagement': 4, 'Deforestation': 2, 'Urbanization': 7,
                              'DamsQuality': 2, 'Siltation': 3, 'Encroachments': 3, 'ineffectiveDisasterPreparedness': 2,
                              'DrainageSystems': 3, 'Coastal vulnerability': 5, 'Landslides': 3, 'watershades': 4,
                              'deteriorating infrastructure': 3, 'population score': 6, 'WetlandLoss': 2, 'AgriculturalPractices': 3,
                              'InadequatePlanning': 2, 'PoliticalFactors': 1},
            'Australia': {'TopographyDrainage': 7, 'RiverManagement': 5, 'Deforestation': 4, 'Urbanization': 7,
                         'DamsQuality': 3, 'Siltation': 5, 'Encroachments': 4, 'ineffectiveDisasterPreparedness': 3,
                         'DrainageSystems': 4, 'Coastal vulnerability': 7, 'Landslides': 4, 'watershades': 6,
                         'deteriorating infrastructure': 3, 'population score': 6, 'WetlandLoss': 4, 'AgriculturalPractices': 5,
                         'InadequatePlanning': 3, 'PoliticalFactors': 2},
            'Japan': {'TopographyDrainage': 8, 'RiverManagement': 4, 'Deforestation': 2, 'Urbanization': 8,
                     'DamsQuality': 2, 'Siltation': 4, 'Encroachments': 3, 'ineffectiveDisasterPreparedness': 2,
                     'DrainageSystems': 2, 'Coastal vulnerability': 8, 'Landslides': 7, 'watershades': 6,
                     'deteriorating infrastructure': 2, 'population score': 6, 'WetlandLoss': 2, 'AgriculturalPractices': 3,
                     'InadequatePlanning': 2, 'PoliticalFactors': 1},
        }
        
        # Get base profile by country, fallback to region-based estimates
        base_profile = None
        if country in regional_profiles:
            base_profile = regional_profiles[country]
        else:
            # Fallback: estimate based on latitude (tropical = higher risk)
            is_tropical = -23.5 <= latitude <= 23.5
            risk_level = 7 if is_tropical else 5
            base_profile = {
                'TopographyDrainage': risk_level, 'RiverManagement': risk_level - 2,
                'Deforestation': risk_level - 1, 'Urbanization': risk_level,
                'DamsQuality': risk_level - 2, 'Siltation': risk_level,
                'Encroachments': risk_level - 1, 'ineffectiveDisasterPreparedness': risk_level - 1,
                'DrainageSystems': risk_level - 2, 'Coastal vulnerability': 8 if abs(latitude) < 30 else 3,
                'Landslides': risk_level, 'watershades': risk_level,
                'deteriorating infrastructure': risk_level - 1, 'population score': risk_level,
                'WetlandLoss': risk_level - 1, 'AgriculturalPractices': risk_level - 2,
                'InadequatePlanning': risk_level - 1, 'PoliticalFactors': risk_level - 2
            }
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'city': f'{city_name}, {country}' if country else city_name,
            'rainfall_mm': round(rainfall_mm, 1),
            'monsoonIntensity': monsoon_intensity,
            'ClimateChange': climate_change_score,
            'clouds_percent': clouds,
            'temperature': temperature,
            'humidity': humidity,
            'description': description,
            'timestamp': timestamp,
            'all_features': base_profile  # All 18 other features auto-suggested
        })
    
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout. Please try again.'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


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
    probabilities = model.predict_proba(X)[0]
    
    # Map predicted class to high/low risk
    risk_label = 'High Risk' if str(prediction).strip().lower() == 'high' else 'Low Risk'
    
    # Get confidence: find probability of predicted class
    # Classes are ordered as they appear in model.classes_
    class_index = list(model.classes_).index(prediction)
    confidence = round(probabilities[class_index] * 100, 1)

    return render_template(
        'index.html',
        feature_names=FEATURE_NAMES,
        prediction=risk_label,
        confidence=confidence,
        input_data=request.form,
    )


@app.route('/predict_scenario', methods=['POST'])
def predict_scenario():
    """
    Predict flood risk based on rainfall amount and region.
    Expects JSON: {"rainfall_mm": 100, "region": "Bangladesh"}
    Returns: {"risk_class": "high", "risk_score": 0.75, "confidence": 85, "message": "..."}
    """
    try:
        data = request.get_json()
        rainfall_mm = float(data.get('rainfall_mm', 0))
        region = data.get('region', 'Bangladesh')
        
        # Validate rainfall range
        if rainfall_mm < 0 or rainfall_mm > 300:
            return jsonify({'error': 'Rainfall must be between 0 and 300 mm'}), 400
        
        # Generate features from rainfall
        features_dict = rainfall_to_features(rainfall_mm, region)
        
        # Extract values in FEATURE_NAMES order for model prediction
        feature_values = [features_dict[name] for name in FEATURE_NAMES]
        X = np.array(feature_values).reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Get confidence score
        class_index = list(model.classes_).index(prediction)
        confidence = round(probabilities[class_index] * 100, 1)
        
        # Get max probability as risk score
        risk_score = round(np.max(probabilities), 3)
        
        # Determine risk message
        risk_messages = {
            'high': '⚠️ High flood risk detected. Immediate action recommended.',
            'medium': '⚡ Moderate flood risk. Preventive measures advised.',
            'low': '✓ Low flood risk. Normal conditions.'
        }
        
        message = risk_messages.get(str(prediction).strip().lower(), 'Risk level unclear.')
        
        return jsonify({
            'success': True,
            'rainfall_mm': rainfall_mm,
            'region': region,
            'risk_class': str(prediction).strip().lower(),
            'risk_score': risk_score,
            'confidence': confidence,
            'message': message,
            'affected_features': {
                'monsoonIntensity': features_dict['monsoonIntensity'],
                'RiverManagement': features_dict['RiverManagement'],
                'Siltation': features_dict['Siltation'],
                'DrainageSystems': features_dict['DrainageSystems'],
                'Coastal vulnerability': features_dict['Coastal vulnerability'],
            }
        })
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)