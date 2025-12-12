#Import Library
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#data loading
df = pd.read_csv(r'C:\Users\ganga\OneDrive\Documents\land_prices.csv')

#print(df.info())

#feature engineering and data cleaning

df['price_per_cent'] = df['price_lakhs'] / df['land_area_cents']
#print(df.info())
#print(df.head())
#print(df.columns)
numeric_features = ['land_area_cents', 'distance_to_school_km',
       'distance_to_airport_km', 'distance_to_railway_station_km',
       'distance_to_hospital_km', 'distance_to_medical_college_km',
       'distance_to_bus_stop_km', 'distance_to_market_km']

#Category handling - one hot encoding
df_changed = pd.get_dummies(df, columns=['location_name','taluk','village','land_type'], drop_first=True)
#print(df_changed.head())

dummy_features =[]
for column_name in df_changed.columns:
    if column_name.startswith('location_name') or column_name.startswith('taluk') or column_name.startswith('village') or column_name.startswith('land_type'):
        dummy_features.append(column_name)
print(dummy_features)

Final_features = numeric_features + dummy_features
#print(Final_features)

#Define features and target
X = df_changed[Final_features]
y = df_changed['price_per_cent']

#split the data
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 42)

#Selection of model
model = RandomForestRegressor(n_estimators=100,random_state=42)

#model training
model.fit(X_train,y_train)

#Evaluate
y_predicted = model.predict(X_test)

df_predicted = pd.DataFrame(
    {
        'Actual': y_test,
        'Predicted': y_predicted
    }
)

print(df_predicted)

#predict with a new record
new_features = model.feature_names_in_.tolist()
#print(new_features)

new_land_details = {
 'land_area_cents':[5.5],
 'distance_to_school_km':[0.8],
 'distance_to_airport_km':[26],
 'distance_to_railway_station_km':[5.1],
 'distance_to_hospital_km':[1.2],
 'distance_to_medical_college_km':[2.5],
 'distance_to_bus_stop_km':[0.4],
 'distance_to_market_km':[1.1],
 'location_name_Beypore':[0],
 'location_name_Chevayur':[1],
 'location_name_Elathur':[0],
 'location_name_Feroke':[0],
 'location_name_Kakkodi':[0],
 'location_name_Karaparamba':[0],
 'location_name_Koduvally':[0],
 'location_name_Koyilandy':[0],
 'location_name_Kunnamangalam':[0],
 'location_name_Mavoor':[0],
 'location_name_Medical College Area':[0],
 'location_name_Olavanna':[0],
 'location_name_Pantheerankavu':[0],
 'location_name_Peruvayal':[0],
 'location_name_Ramanattukara':[0],
 'location_name_Thamarassery':[0],
 'location_name_Vatakara':[0],
 'taluk_Kozhikode':[1],
 'taluk_Thamarassery':[0],
 'taluk_Vatakara':[0],
 'village_Beypore':[0],
 'village_Chevayur':[1],
 'village_Elathur':[0],
 'village_Feroke':[0],
 'village_Kakkodi':[0],
 'village_Karaparamba':[0],
 'village_Koduvally':[0],
 'village_Koyilandy':[0],
 'village_Kozhikode':[0],
 'village_Kunnamangalam':[0],
 'village_Mavoor':[0],
 'village_Olavanna':[0],
 'village_Pantheerankavu':[0],
 'village_Peruvayal':[0],
 'village_Ramanattukara':[0],
 'village_Thamarassery':[0],
 'village_Vatakara':[0],
 'land_type_Commercial':[0],
 'land_type_Residential':[1]
}
new_land_df = pd.DataFrame(new_land_details)
#prediction for new value
new_land_predicted_value = model.predict(new_land_df)
print("Predicted value for the new data:",new_land_predicted_value)
