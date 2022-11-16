import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from pmdarima import auto_arima
import statsmodels.api as sm
import pickle


df = pd.read_csv("train.csv", header=[0,1], low_memory=False)
df.fillna(0, inplace=True)


X = df.drop(labels=['price'], axis=1)
Y = df['price']

stepwise_fit = auto_arima(df['price'], trace=True, suppress_warning=True)
stepwise_fit.summary()

# ARIMA(1,1,0)(0,0,0)[0]

tss = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

y_train.rename(columns={'Unnamed: 210_level_1': 'price'}, inplace=True)
y_test.rename(columns={'Unnamed: 210_level_1': 'price'}, inplace=True)

df_c = pd.concat([X_train, y_train], axis=1)


model = sm.tsa.arima.ARIMA(df_c, order=(1,1,0))
model = model.fit()

# load / store model into the pickle binary file
#
# with open('model_arima', 'wb') as f:
#     pickle.dump(model, f)
#
# with open('model_arima', 'rb') as f:
#     model = pickle.load(f)
