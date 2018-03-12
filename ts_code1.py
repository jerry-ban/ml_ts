
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
print(os.getcwd())

shampoo_file= 'sales-of-shampoo-over-a-three-ye.csv'
def date_col_parser(x):
	return pd.datetime.strptime('190'+x, '%Y-%m')

def load_data():
    from pandas.compat import StringIO
    data = pd.read_csv(shampoo_file, names =["Month","Sales"], header=0, parse_dates=["Month"], index_col=0, squeeze=True, date_parser=date_col_parser)
    #data = pd.read_csv(shampoo_file, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=date_col_parser)
    # summarize first few rows
    print(data.head())
    # line plot
    data.plot()
    plt.show()
    return data


sale_series = load_data()
X = sale_series.values
cycle = 12
train, test = X[0:-cycle], X[-cycle:]

def base_line_prediction(train, test):
    history =[x for x in train]
    predictions = []
    for i in range(len(test)):
        predictions.append(history[-1])
        history.append(test[i])
    rmse = math.sqrt(skmetrics.mean_squared_error(test, predictions))
    title= "RMSE: %.3f" % rmse
    print(title)
    plt.figure()
    plt.plot(test)
    plt.plot(predictions)
    plt.title("baseline(RMSE:{})".format(title))
    plt.show()

base_line_prediction(train, test)

### LSTM data preprocess
# transform TS to supervised learning
def ts_to_supervisesd(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df_result=pd.concat(columns, axis=1)
    df_result.fillna(0, inplace=True)
    return df_result

supervised_df = ts_to_supervisesd(X)

def difference(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value=data[i] - data[i-interval]
        diff.append(value)
    return pd.Series(diff)

def inverse_diff(history, yhat, interval=1):
    return yhat + history[-interval]

def invert_data(history, diference, interval = 1):
    inverted = list()
    for i in range(len(differenced)):
        value = inverse_diff(history, differenced[i], len(history)-i)
        inverted.append(value)
    result = pd.Series(inverted)
    return result

differenced = difference(sale_series, 1)
inverted = invert_data(sale_series, differenced, 1)

differenced.head(5)
inverted.head(5)
