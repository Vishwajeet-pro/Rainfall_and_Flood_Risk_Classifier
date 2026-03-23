# Rainfall and Flood Risk Classifier - Line-by-Line Explanation
*For complete beginners. Each Python line explained like you're 5. No prior knowledge needed. We use simple words, examples, and fun analogies.*

## What is this Project?
This is a **web app** that predicts if an area has **high or low flood risk** based on 20 factors (like rain strength, city growth). 
- Uses **Machine Learning (ML)**: Computer learns from past flood data to guess future risks. Like teaching a kid patterns from examples.
- Files:
  - `model_train.py`: Teaches the computer (trains model).
  - `app.py`: Runs a website to ask for predictions.
  - `get_insight.py`: Shows which factors matter most.

## Before Reading: Quick Setup (Don't worry if you skip)
1. Install Python (free from python.org).
2. Open terminal/Command Prompt.
3. Run: `pip install pandas scikit-learn flask numpy openpyxl` (gets tools).
4. Put `data/flood.csv` in folder (data file with flood info).

Now, line-by-line explanations!

## 1. model_train.py - Training the "Brain" (Model)
This file reads flood data, prepares it, trains a ML model, tests it, and saves it.

```
import pandas as pd 
```
- **What:** Brings in "pandas" library (like Excel for Python).
- **pd:** Short nickname (alias) for pandas, saves typing.
- **Why:** To read/load data from CSV file easily.
- **Concept:** Libraries are pre-made toolkits. `import` = "give me these tools". Analogy: Borrowing books from library.

```
from sklearn.model_selection import train_test_split
```
- **What:** Gets `train_test_split` from scikit-learn (ML toolkit).
- **Why:** Splits data into "practice" (train) and "test" sets. Like studying with some questions, testing with others.
- **Concept:** `from A import B`: Gets specific tool B from toolbox A.

```
from sklearn.ensemble import RandomForestClassifier 
```
- **What:** Gets RandomForestClassifier (smart ML model).
- **Why:** Random Forest is many "decision trees" voting on answer. Great for risk prediction.
- **Concept:** Classifier = sorts things into categories (low/medium/high risk).

```
from sklearn.metrics import accuracy_score
```
- **What:** Tool to check how good model is (% correct).
- **Why:** Measure accuracy, like test score.

```
import pickle 
```
- **What:** Library to save/load Python objects (like model).
- **Why:** Save trained model to file, reuse later.

```
# 1.Load the data
df = pd.read_csv('data/flood.csv')
```
- **What:** `df` = DataFrame (table like Excel). Loads CSV data.
- **Why:** Starting point - read flood data.
- **Concept:** CSV = comma-separated file. `df.head()` shows first rows.

```
# 2.Define your input features 
X = df.drop('FloodProbability', axis=1)  
```
- **What:** `X` = inputs (20 factors, drops 'FloodProbability' column).
- **axis=1:** Columns (axis=0 is rows).
- **Why:** ML needs inputs (X) separate from output (y/target).
- **Analogy:** X = clues, drop answer column.

```
# Convert continuous probability into discrete classes for classification
# Adjust thresholds to your business logic (low/medium/high risk)
y = pd.cut(df['FloodProbability'], bins=[-0.01, 0.33, 0.66, 1.01], labels=['low', 'medium', 'high'])
```
- **What:** `y` = outputs. `pd.cut` bins numbers into groups.
  - bins: -0.01 to 0.33=low, 0.33-0.66=medium, 0.66-1.01=high.
- **Why:** Model predicts categories, not numbers.
- **Concept:** Binning = grouping ranges (like age: kid/teen/adult).

```
# Split data 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **What:** Splits 80% train, 20% test. `random_state=42` = same split always.
- **Why:** Train on 80%, test on unseen 20% for real accuracy.

```
# Train a classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
print("Training the model ... please wait ...")
model.fit(X_train, y_train)
```
- **What:** Creates model with 50 trees. `fit` = train/learn.
- **Why:** Computer studies patterns.

```
# check Quality of the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```
- **What:** `predict` = guess on test data. accuracy_score = % correct.
- **f-string:** `f"Text {var}"` inserts value.

```
# save the model
with open('flood_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Success! 'flood_model.pkl' has been created.")
```
- **What:** `with open` safely opens file. 'wb'=write binary. `dump` saves model.
- **Why:** Reuse without retraining.
- **Concept:** Pickle = Python's save-anything format.

## 2. app.py - The Website (Flask App)
Runs web server. Users input factors, get prediction.

```
from flask import Flask, render_template, request
```
- **Flask:** Web framework (easy websites).
- **render_template:** Loads HTML files.
- **request:** Gets form data.

```
import pickle
import numpy as np
```
- **numpy (np):** Math arrays, fast numbers.

```
app = Flask(__name__)
```
- **What:** Creates app. `__name__` = current file/module.

```
FEATURE_NAMES = [
    'monsoonIntensity', 'TopographyDrainage', ... (20 factors)
]
```
- **List:** Hardcoded features (matches data).

```
# Load trained model
with open('flood_model.pkl', 'rb') as f:
    model = pickle.load(f)
```
- **'rb'**: Read binary. `load` = unpickle.

```
@app.route('/')
def index():
    return render_template('index.html', feature_names=FEATURE_NAMES)
```
- **@app.route('/')**: URL "/" maps to `index()` function.
- **Why:** Homepage shows form with feature names.

```
@app.route('/predict', methods=['POST'])
def predict():
```
- **'/predict', POST**: Handles form submit.

```
    try:
        values = [float(request.form[name]) for name in FEATURE_NAMES]
    except (KeyError, ValueError):
        return render_template(
            'index.html',
            feature_names=FEATURE_NAMES,
            error='Please provide numeric values for all fields.',
        )
```
- **try/except:** Try code, catch errors.
- **request.form[name]**: Gets input value.
- **list comprehension:** `[float(val) for name in list]` = loop shorter.
- **float()**: String to number.

```
    X = np.array(values).reshape(1, -1)
```
- **np.array**: Number array. reshape(1,-1): 1 row, any columns.

```
    prediction = model.predict(X)[0]
```
- **predict(X)[0]**: Get first (only) prediction.

```
    risk_label = 'High Risk' if str(prediction).strip().lower() == 'high' else 'Low Risk'
```
- **if/else:** Check if 'high', else 'Low Risk'.
- **str().strip().lower()**: Clean string.

```
    return render_template(
        'index.html',
        feature_names=FEATURE_NAMES,
        prediction=risk_label,
        input_data=request.form,
    )
```
- **Pass vars to HTML** (shows result).

```
if __name__ == '__main__':
    app.run(debug=True)
```
- **What:** Runs server if file run directly (`python app.py`).
- **debug=True:** Auto-reload on changes.

## 3. get_insight.py - What Matters Most?
Shows feature importance (which factors cause most floods).

```
import pandas as pd 
import pickle
import os 
```
- **os:** File/OS stuff (not used here).

```
with open('flood_model.pkl', 'rb') as f:
    model = pickle.load(f)
```
- Load model (same as app).

```
df = pd.read_csv('data/flood.csv')
```
- Load data.

```
# get feature name
features = [
    'monsoonIntensity', ... (20 names)
]
```
- Hardcoded feature list.

```
print("DataFrame columns:", df.columns.tolist())
print("Features used:", features)
```
- **print**: Outputs to screen.
- **df.columns.tolist()**: List of column names.

```
# Create a dataframe oof importance 
importance_df = pd.DataFrame({'Factor':features,'Importance':model.feature_importances_}).sort_values(by='Importance',ascending=False)
```
- **pd.DataFrame**: New table from dict.
- **model.feature_importances_**: RandomForest scores (0-1, higher=more important).
- **sort_values**: Sort descending.

```
importance_df.to_csv('feature_importance.csv',index=False)
print("Feature importance has been saved to 'feature_importance.csv'.")
```
- Save to CSV, no index column.

## Glossary (Key Words)
- **DataFrame**: Table (rows/columns).
- **fit()**: Train model.
- **predict()**: Use model.
- **Route**: Web URL handler.
- **Pickle**: Save Python stuff to file.
- **Array**: List of numbers.
- **Library**: Reusable code pack.

## Run It!
1. `python model_train.py` (train).
2. `python app.py` (website: http://127.0.0.1:5000).
3. `python get_insight.py` (importance CSV).

All done! Open in browser/editor. Questions? Reread lines 😊.
