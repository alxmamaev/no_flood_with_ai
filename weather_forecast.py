import requests
import pandas as pd
import numpy as np
from metadata import stantion_coord
from tqdm import tqdm

KEY = "f14b9bc97b1340f48f5164047202211"


def get_weather_foreсast_by_coord(lat, lon, start_date):
    forecast = []
    for i in range(1, 11):
        date = start_date + pd.Timedelta(days=i)

        url = "http://api.weatherapi.com/v1/history.json?key={}&q={}, {}&dt={}".format(KEY, lat, lon, 
                                                                                       date.strftime("%Y-%m-%d"))
        response = requests.get(url)
        if response.status_code == 400:
            break  

        _forecast = response.json()["forecast"]["forecastday"][0]
        _forecast["day"]["daily_chance_of_rain"] = float(any([i["will_it_rain"] for i in _forecast["hour"]])) * 100
        _forecast["day"]["daily_chance_of_snow"] = float(any([i["will_it_snow"] for i in _forecast["hour"]])) * 100
        forecast.append(_forecast)

    if len(forecast) != 10:
        url = "http://api.weatherapi.com/v1/forecast.json?key={}&q={}, {}&days=15".format(KEY, lat, lon)
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception("Status code is {}".format(response.status_code))
            
        forecast += response.json()["forecast"]["forecastday"]
    
    data = pd.DataFrame()
    data["time"] = pd.to_datetime([i["date"] for i in forecast])
    data["temperature_air"] = [i["day"]["avgtemp_c"] for i in forecast]
    data["humidity"] = [i["day"]["avghumidity"] for i in forecast]
    data["precipitation_amount"] = [i["day"]["totalprecip_mm"] for i in forecast]
    data["pressure"] = [np.mean([j["pressure_mb"] for j in i["hour"]]) for i in forecast]
    data["wind_direction"] = [np.mean([j["wind_degree"] for j in i["hour"]]) for i in forecast]
    data["wind_speed"] = [np.mean([j["wind_kph"] for j in i["hour"]]) for i in forecast]
    data["wind_speed"] = data["wind_speed"] * 1000/3600
    data["rain"] = [float(float(i["day"]["daily_chance_of_rain"]) > 50) for i in forecast]
    data["snow"] = [float(float(i["day"]["daily_chance_of_snow"]) > 50) for i in forecast]
    
    return data


def get_weather_foreсast(start_date):
    full_forecast = []
    for stantion_id, coord in tqdm(stantion_coord.items()):
        forecast = get_weather_foreсast_by_coord(coord["lat"], coord["lon"], start_date)
        forecast["identifier"] = stantion_id
        full_forecast.append(forecast)

    full_forecast = pd.concat(full_forecast)
    full_forecast = full_forecast.set_index(["identifier", "time"])

    full_forecast = full_forecast.sort_index()
    return full_forecast
    