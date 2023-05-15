from tsp import TimeSeries
import pandas as pd


df = pd.read_csv('./mini_data/tsp_data/AirPassengers.csv')
df["Month"] = pd.to_datetime(df["Month"])

timeseries = TimeSeries.from_df(df, label_cols='#Passengers', time_col="Month")
print(timeseries)

