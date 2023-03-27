import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


# Veri setini yükleme
df = pd.read_csv('Data/airline_passengers.csv')

df['ds']=pd.DatetimeIndex(df['Month']+'-01')

df.drop(['Month'],axis=1,inplace=True)

df.columns = ['y','ds']

print(df.head(5))

m = Prophet(interval_width=0.95,daily_seasonality=True,seasonality_prior_scale=20)

model = m.fit(df)

future = m.make_future_dataframe(periods=100,freq='D',include_history=True)
forecast = m.predict(future)
print(forecast.head())

m.plot(forecast)

plt.show()