import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import pickle 

# 1.Load the data
df = pd.read_csv('data/flood.csv')

# 2.Define your input features 
X = df.drop('FloodProbability', axis=1)  

# Convert continuous probability into discrete classes for classification
# Adjust thresholds to your business logic (low/medium/high risk)
y = pd.cut(df['FloodProbability'], bins=[-0.01, 0.33, 0.66, 1.01], labels=['low', 'medium', 'high'])

# Split data 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
print("Training the model ... please wait ...")
model.fit(X_train, y_train)

# check Quality of the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# save the model
with open('flood_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Success! 'flood_model.pkl' has been created.")




