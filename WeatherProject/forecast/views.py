from django.shortcuts import render

# Create your views here.

# from django.http import HttpResponse

# Section 1: Import Libraries
import requests #This library helps us to fetch data from API
import pandas as pd #This library helps us to handle and analyse data
import numpy as np #This library helps us to handle numerical operations
from sklearn.model_selection import train_test_split #This library helps us to split data into training and testing sets
from sklearn.preprocessing import LabelEncoder #This library is used to convert catogorical data into numerical values
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # These are ML models for classification and regression tasks
from sklearn.metrics import mean_squared_error #This function measures the accuracy of our predictions
from datetime import datetime, timedelta #These functions handles dates and times
import pytz #For different timezones
import os #For csv files readings

API_KEY = '89cc6721303ae8c74d8d43f09b51e77a'  #API_KEY (variable) is replaced by actual API key (Large Number) present in RHS
BASE_URL = 'https://api.openweathermap.org/data/2.5/' # BASE_URL (variable) is used for making API requests



# 1. Fetch Current Weather Data
def get_current_weather(city): #Gets weather for a specific city
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric" #Constructs the API request URL
    response = requests.get(url) #Sends the get requests to weather API and get weather for city we passed in URL
    data = response.json() #Converts received response in json format for working easily with the data
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country' : data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'Pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed'],

        'clouds': data['clouds']['all'],
        'visibility': data['visibility'],
    }



# 2. Read Historical Data
def read_historical_data(filename): #Reads historical data for a specific csv file
    df = pd.read_csv(filename) #Loads csv file into dataframe
    df = df.dropna() #Removes rows with missing values
    df = df.drop_duplicates() #Removes rows with duplicate values
    return df



# 3. Prepare data for training
def prepare_data(data):
    le = LabelEncoder() #Create a LabelEncoder instance
    data['WindGustDir'] = le.fit_transform(data['WindGustDir']) #Encode 'WindGustDir' (categorical/string) data into numerical values
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow']) #Encode 'RainTomorrow' (categorical/string) data into numerical values

    #Define the feature variables and target variables
    X = data.loc[:, ['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']] #Feature Variables
    y = data['RainTomorrow'] #Target Variable

    return X, y, le



# 4. Train Rain Prediction Model
def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Split data into training and testing sets
    model = RandomForestClassifier(n_estimators=100, random_state=42) #Create a Random Forest Classifier model
    model.fit(X_train, y_train) #Train the model using the training data

    y_pred = model.predict(X_test) #Make predictions on the test data
    #accuracy = accuracy_score(y_test, y_pred) #Calculate the accuracy of the model

    print("mean Squared Error for Rain Model")

    print(mean_squared_error(y_test, y_pred))

    return model



# 5. Prepare Regression Data
def prepare_regression_data(data, feature):
    X, y = [], [] #Initialize list for feature values and target values

    for i in range(len(data)-1):
        X.append(data[feature].iloc[i]) #Append feature values to X
        y.append(data[feature].iloc[i+1]) #Append target value to y

    X = np.array(X).reshape(-1, 1) #Reshape X into a 2D array #Format of regression model
    y = np.array(y) #Convert y to a numpy array

    return X, y



# Train Regression Model
def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42) #Create a Random Forest Regressor model
    model.fit(X, y) #Train the model using the provided data

    return model



# Predict Future
def predict_future(model, current_value):
    predictions = [current_value]

    for i in range(5):
        next_value = model.predict(np.array([predictions[-1]]).reshape(1,-1))

        predictions.append(next_value[0])

    return predictions[1:]



# Weather Analysis Functions
    # return HttpResponse('<h1>Weather Prediction App</h1>')
def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        current_weather = get_current_weather(city)

        #Load Historical Data
        csv_path = os.path.join('D:\\MLWebAppProject\\weather.csv')
        historical_data = read_historical_data(csv_path)

        #Prepare and train the rain prediction model
        X, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(X, y)

        #Mapping wind directions to compass points
        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N",0,11.25), ("NNE",11.25,33.75), ("NE",33.75,56.25),
            ("ENE",56.25,78.75), ("E",78.75,101.25), ("ESE",101.25,123.75),
            ("SE",123.75,146.25), ("SSE",146.25,168.75), ("S",168.75,191.25),
            ("SSW",191.25,213.75), ("SW",213.75,236.25), ("WSW",236.25,258.75),
            ("W",258.75,281.25), ("WNW",281.25,303.75), ("NW",303.75,326.25),
            ("NNW",326.25,348.75)
        ]
        compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather['Wind_Gust_Speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['Pressure'],
            'Temp': current_weather['current_temp']
        }

        current_df = pd.DataFrame([current_data])

        #Rain Prediction
        rain_prediction = rain_model.predict(current_df)[0]
        #rain_prediction_label = le.inverse_transform(rain_prediction)[0]

        #Prepare regression model for temperature and humidity
        X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
        temp_model = train_regression_model(X_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)

        #Predict future temperature and humidity
        future_temp = predict_future(temp_model, current_weather['temp_min'])
        future_humidity = predict_future(hum_model, current_weather['humidity'])

        #Prepare time for future predictions
        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

        future_times = [(next_hour + timedelta(hours=i)).strftime('%H:00') for i in range(5)]

        #Store each variable value in a separately
        time1, time2,time3, time4, time5 = future_times
        temp1, temp2,temp3, temp4, temp5 = future_temp
        hum1, hum2,hum3, hum4, hum5 = future_humidity

        #Passing data to the template
        context = {
           'location': city,
           'current_temp': current_weather['current_temp'],
           'MinTemp': current_weather['temp_min'],
           'MaxTemp': current_weather['temp_max'],
           'feels_like': current_weather['feels_like'],
           'humidity': current_weather['humidity'],
           'clouds': current_weather['clouds'],
           'description': current_weather['description'],
           'city': current_weather['city'],
           'country': current_weather['country'],

           'time': datetime.now(),
           'date': datetime.now().strftime("%B %d, %Y"),

           'wind': current_weather['Wind_Gust_Speed'],
           'pressure': current_weather['Pressure'],
           'visibility': current_weather['visibility'],

           'time1': time1,
           'time2': time2,
           'time3': time3,
           'time4': time4,
           'time5': time5,

           'temp1': f"{round(temp1,1)}",
           'temp2': f"{round(temp2,1)}",
           'temp3': f"{round(temp3,1)}",
           'temp4': f"{round(temp4,1)}",
           'temp5': f"{round(temp5,1)}",

           'hum1': f"{round(hum1,1)}",
           'hum2': f"{round(hum2,1)}",
           'hum3': f"{round(hum3,1)}",
           'hum4': f"{round(hum4,1)}",
           'hum5': f"{round(hum5,1)}",
        }

        return render(request, 'weather.html', context)
    
    return render(request, 'weather.html')
            
