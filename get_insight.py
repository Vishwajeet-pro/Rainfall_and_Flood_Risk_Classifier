import pandas as pd 
import pickle
import os 

with open('flood_model.pkl', 'rb') as f:
    model = pickle.load(f)
df = pd.read_csv('data/flood.csv')


# get feature name

features = [
    'monsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
    'AgriculturalPractices', 'Encroachments', 'ineffectiveDisasterPreparedness',
    'DrainageSystems', 'Coastal vulnerability', 'Landslides', 'watershades',
    'deteriorating infrastructure', 'population score', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]
print("DataFrame columns:", df.columns.tolist())
print("Features used:", features)

# Create a dataframe oof importance 

importance_df = pd.DataFrame({'Factor':features,'Importance':model.feature_importances_}).sort_values(by='Importance',ascending=False)



importance_df.to_csv('feature_importance.csv',index=False)
print("Feature importance has been saved to 'feature_importance.csv'.")