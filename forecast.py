import argparse
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.meteo import load_meteo
from metadata import target_station_ids, meteo_ids, nearest_stantions, nearest_meteo
from weather_forecast import get_weather_foreсast
from functools import partial
from utils.meteo import calculate_meteo
from pandarallel import pandarallel
from features_geneartion import process_data
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

pandarallel.initialize(progress_bar=False)
# Тут говнокод, но что уж поделать

parser = argparse.ArgumentParser()
parser.add_argument("f_day", type=str)
parser.add_argument("l_day", type=str)
p = parser.parse_args()

f_day = pd.to_datetime(p.f_day)
l_day = pd.to_datetime(p.l_day)

start_date = f_day - pd.Timedelta(days=360*4)
end_date = l_day + pd.Timedelta(days=15)

print("[1] Read whater level history dataset")
dataset = pd.read_csv("datasets/hydro_2018-2020/new_data_all.csv", sep=";")
dataset["time"] = pd.to_datetime(dataset["time"])
dataset = dataset.sort_index()

def add_missing_dates(data, identifier):
    data = data.set_index("time").sort_index()
    data = data.asfreq("D")
    data = data.reset_index()
    data["identifier"] = identifier
    
    return data


dataset = dataset.set_index("identifier")
dataset = pd.concat([add_missing_dates(dataset.loc[identifier], identifier) 
                                       for identifier in dataset.index.unique()])

dataset = dataset.rename(columns={"max_level": "level"})
dataset = dataset.set_index(["identifier", "time"])
dataset = dataset.sort_index()


print("[2] Add near stantions levels")

new_dataset = []
for stantion_id in target_station_ids:
    d = dataset.loc[stantion_id].copy()
    d = d.sort_index()
    d["identifier"] = stantion_id
    
    nst = nearest_stantions[stantion_id]
    
    for i in range(3):
        if len(nst) > i:
            d["near_level_{}".format(i+1)] = dataset.loc[nst[i][0]]["level"]
            d["near_dist_{}".format(i+1)] = nst[i][1]
            d["near_dist_{}".format(i+1)][d["near_level_{}".format(i+1)].isna()] = -1
            d["near_level_{}".format(i+1)] = d["near_level_{}".format(i+1)].fillna(-1)
        else:
            d["near_level_{}".format(i+1)] = -1
            d["near_dist_{}".format(i+1)] = -1

    d = d.loc[start_date:end_date]    
    new_dataset.append(d)

dataset = pd.concat(new_dataset).reset_index()
del new_dataset


meteo = {}
print("[3] Load meteo data")
for meteo_id in tqdm(meteo_ids):
    meteo[int(meteo_id)] = load_meteo(meteo_id, start_date=start_date, end_date=end_date).interpolate()

print("[4] Calculate meteo data")
_calculate_meteo = partial(calculate_meteo, meteo, nearest_meteo)
meteo_data = dataset.parallel_apply(_calculate_meteo, axis=1)

dataset = pd.concat((dataset, meteo_data), axis=1)
del meteo_data
max_date = (dataset.groupby("identifier")["time"].max()).min()
dataset = dataset.set_index(["identifier", "time"])

dataset = dataset.interpolate()

if l_day > max_date:
    dataset = dataset.reset_index()

    new_dataset = []
    for stantion_id in target_station_ids:
        d = dataset[dataset["identifier"] == stantion_id]
        d = d[d["time"].between(d.iloc[0]["time"], f_day)]
        new_dataset.append(d)

    dataset = pd.concat(new_dataset)
    del new_dataset
    dataset = dataset.set_index(["identifier", "time"])

    print("[5] Get meteo forecast")
    meteo_forecast = get_weather_foreсast(f_day)
    dataset = pd.concat((dataset, meteo_forecast))
    dataset = dataset.sort_index()

print("[6] Calculcate features")
dataset = process_data(dataset)


print("[7] Load trend prediction model")
model = CatBoostRegressor().load_model("model.catboost")

b = [dataset[i][1].loc[f_day] for i in target_station_ids]
test_dataset = pd.DataFrame([dataset[i][0].loc[f_day] for i in target_station_ids])
test_dataset = test_dataset[model.feature_names_]
test_dataset["month"] = test_dataset["month"].astype(np.int32)
test_dataset["identifier"] = test_dataset["identifier"].astype(np.int32)
k = model.predict(test_dataset)

submit = pd.DataFrame()

print("[8] Making trend")
for i, stantion_id in enumerate(target_station_ids):
    level = []
    for day in range(1, 11):
        level.append(round(b[i] + day * k[i], 2))
    submit["0" + str(stantion_id)] = level

print("[9] Save submit")
submit.index = [f_day + pd.Timedelta(days=i) for i in range(1, 11)]
submit.index.name = "date"
submit.to_csv("datasets/submission.csv")

print("[10] Done!")