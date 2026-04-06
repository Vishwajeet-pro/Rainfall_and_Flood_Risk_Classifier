# Rainfall and Flood Risk Classifier

A Flask-based flood risk prediction application that uses a trained Random Forest model and regional rainfall simulations to estimate flood risk. It also includes a new Risk Reduction Roadmap Generator to recommend prioritized infrastructure and policy improvements.

## Features

- Flood risk prediction based on rainfall scenarios and regional profiles
- Interactive rainfall simulator with real-time model predictions
- Analytics dashboard showing feature importance for the flood risk model
- Roadmap generator that recommends resilience improvements prioritized by impact, cost, and ROI
- Weather-driven auto-suggestion of regional baseline risk factors

## Repository Structure

- `app.py` — Flask application with routes for prediction, simulator, analytics, weather lookup, and roadmap generation
- `roadmap_engine.py` — Roadmap logic with improvement catalog, impact application, ROI ranking, and improved risk calculation
- `model_train.py` — Train the `RandomForestClassifier` and save `flood_model.pkl`
- `get_insight.py` — Extract feature importance from the trained model
- `templates/` — HTML templates for the web UI
- `static/` — Static resources such as images
- `data/flood.csv` — Dataset used for model training
- `flood_model.pkl` — Serialized trained model

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd Rainfall_and_Flood_Risk_Classifier
```

2. Create a Python virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

```powershell
# Windows PowerShell
venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install flask scikit-learn numpy requests
```

> If you want to include additional packages or freeze dependencies, create a `requirements.txt` file.

## Running the App

Start the Flask application:

```powershell
python app.py
```

Open your browser at:

```
http://127.0.0.1:5000
```

## Usage

- `GET /` — Main predictor interface
- `GET /simulator` — Rainfall scenario simulator
- `GET /analytics` — Feature importance analytics dashboard
- `GET /roadmap` — Roadmap generator for flood risk reduction
- `POST /predict_scenario` — JSON endpoint for rainfall-based predictions
- `POST /generate_roadmap` — JSON endpoint for roadmap recommendations
- `POST /get_weather` — Fetch weather data and auto-suggest regional risk factors

## Roadmap Generator

The roadmap feature helps users select resilience improvements such as:

- early warning systems
- drainage upgrades
- river dredging
- wetland restoration
- dam strengthening
- anti-encroachment enforcement
- community training
- governance reform

It computes before/after risk scores, estimated costs, timelines, and ROI-based ranking.

## Demo 

<img width="650" height="527" alt="Screenshot 2026-04-06 085657" src="https://github.com/user-attachments/assets/023d7c6d-ef79-4122-a9a3-e7343c756de0" />
<img width="652" height="577" alt="Screenshot 2026-04-06 085742" src="https://github.com/user-attachments/assets/7fd6677c-341f-417e-84a6-b2d447ca53b5" />

<br><br>
<br><br>

<img width="616" height="584" alt="Screenshot 2026-04-06 085907" src="https://github.com/user-attachments/assets/7d3c6a3d-727d-4fcc-9f98-8fdf2ce5d32b" />
<br><br>
<br><br>

<img width="528" height="535" alt="Screenshot 2026-04-06 085951" src="https://github.com/user-attachments/assets/b1079d3a-8fb3-4d86-91a2-d69693f5af76" />
<br><br>
<br><br>
<img width="627" height="318" alt="Screenshot 2026-04-06 090107" src="https://github.com/user-attachments/assets/06dfa99c-3b0c-4360-8a05-f50f50dfe9ef" />


## Notes

- The model uses a classification target derived from `FloodProbability` in `data/flood.csv`.
- All feature inputs are expected on a `0-14` scale.
- The app is intended for demonstration and educational purposes.

## License

This project is available to use and modify freely.
