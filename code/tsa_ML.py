import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_absolute_error, mean_squared_error

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set random seed for reproducibility
np.random.seed(42)
cal = USFederalHolidayCalendar()

path = 'Electric_Production_tm.csv'
path = 'simulation.csv'

#########################
DF = pd.read_csv(path)
DF.columns = ['date', 'sales']
DF.dtypes


######################
plt.figure(figsize=(14, 7))
plt.plot(DF['date'], DF['sales'], label='Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Data')
plt.legend()
plt.show()

c = ['date', 'sales']
DF = DF.sort_values(['date'])

def create_date(DF, x, lag_n):
    DF1 = DF.copy()
    DF1['date'] = pd.to_datetime(DF1['date'])
    DF1['weekday'] = DF1['date'].dt.day_name()
  
    DF1['month'] = DF1['date'].dt.month_name()
    dummies_wkd = pd.get_dummies(DF1['weekday'], prefix='is') + 0
    DF1 = pd.concat([DF1, dummies_wkd], axis=1)
    
    dummies_mon = pd.get_dummies(DF1['month'], prefix='is') + 0
    DF1 = pd.concat([DF1, dummies_mon], axis=1)
    
    holidays = cal.holidays(start=DF1['date'].min(), end=DF1['date'].max())
    DF1['is_holiday'] = DF1['date'].isin(holidays).astype(int)
    DF1['is_holiday'].mean()
    
    DF1['t'] = (DF1['date'] - DF1['date'].min()).dt.days
    DF1['t_lg'] = np.log((DF1['t'] + 1))

    for lag in range(1, lag_n + 1):
        DF1[f'lag{lag}'] = DF1[x].shift(lag)
        
    DF1['rolling_mean_7'] = DF1[x].rolling(window=7).mean()
    DF1['rolling_std_7'] = DF1[x].rolling(window=7).std()
    DF1['rolling_mean_30'] = DF1[x].rolling(window=30).mean()
    DF1['rolling_std_30'] = DF1[x].rolling(window=30).std()

    DF1 = DF1.drop(['month','weekday'], axis = 1)
    DF1 = DF1.fillna(0)
   
    return DF1

new_DF = create_date(DF, 'sales', 3) 
new_DF = new_DF.sort_values(['date'])

max_y = new_DF['sales'].max()
def transform_target(y, transform_fun):
    if transform_fun == 'log':
        return np.log(y)
    elif transform_fun == 'sqrt':
        return np.sqrt(y)
    elif transform_fun == 'logistic':
        return 1 / (1 + np.exp(-y/max_y))
    else:
        raise ValueError("Unsupported transformation function. Choose from 'log', 'sqrt', 'logistic'.")

def inverse_transform_target(y, transform_fun):
    if transform_fun == 'log':
        return np.exp(y)
    elif transform_fun == 'sqrt':
        return np.square(y)
    elif transform_fun == 'logistic':
        return -np.log((1 / y) - 1)*max_y
    else:
        raise ValueError("Unsupported transformation function. Choose from 'log', 'sqrt', 'logistic'.")


def train_ml_model(X_train, y_train, X_valid, y_valid, model_type='lightgbm', transform_fun=None):
    if transform_fun is not None:
        y_train = transform_target(y_train, transform_fun)
        y_valid = transform_target(y_valid, transform_fun)
    
    if model_type == 'lightgbm':
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1
        }

        model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[valid_data],
            num_boost_round=300,
            callbacks=[
                lgb.early_stopping(stopping_rounds=3),
            ]
        )

    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:linear', n_estimators=150)
        model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid, y_valid)], verbose=False)
  
    elif model_type == 'linear_regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
  
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=150)
        model.fit(X_train, y_train)
    else:
        raise ValueError("Unsupported model type. Choose from 'lightgbm', 'xgboost', 'linear_regression', 'random_forest'.")

    return model


df = new_DF.copy()
features = list(df.columns)
rem = ['date','sales']
for itv in rem:
    features.remove(itv)
    
def getcorr_cut(Y, df, varnamelist, thresh):
    corr = list()
    meanv = list()
    for vname in varnamelist: 
            X = df[vname]              
            C = np.corrcoef(X, Y)      
            beta=np.round(C[1, 0],4)
            corr = corr+[beta] 
            avg = round(X.mean(), 6)
            meanv = meanv + [avg] 
    
    corrdf = pd.DataFrame({'varname': varnamelist, 'correlation': corr, 'mean': meanv})
    corrdf['abscorr'] = np.abs(corrdf['correlation'])
    corrdf.sort_values(['abscorr'], ascending=False, inplace=True)
    seq = range(1,len(corrdf)+1)
    corrdf['order']=seq
    corrdf['meanabs'] = np.abs(corrdf['mean']) 
    corrdf['abscorr'] = corrdf['abscorr'].fillna(0.0)
    corrdf = corrdf[(corrdf.abscorr >= thresh)]
    return corrdf  
   
X = df[features]
y = df['sales']

cor_df = getcorr_cut(y, X, features, 0.05)
features = list(cor_df.varname)
X = df[features]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
model = train_ml_model(X_train, y_train, X_valid, y_valid, model_type='lightgbm', transform_fun=None)

def rolling_window_validation(DF, target_col, lag_n, train_size,\
  test_size, features, model_type='lightgbm', transform_fun=None):
    
    predictions = []
    actuals = []
    test_indices = []
    
    for start in range(0, len(DF) - train_size - test_size + 1, test_size):
        train = DF[start:start + train_size]
        test = DF[start + train_size:start + train_size + test_size]
        test_indices.extend(test.index)
        
        X_train = train.drop(columns=[target_col, 'date'])
        y_train = train[target_col]
        X_test = test.drop(columns=[target_col, 'date'])
        y_test = test[target_col]
        
        X_train = X_train[features]
        X_test = X_test[features]
        
        model = train_ml_model(X_train, y_train, X_test, y_test, model_type=model_type, transform_fun=transform_fun)
        
        y_pred = model.predict(X_test)
        if transform_fun is not None:
            y_pred = inverse_transform_target(y_pred, transform_fun)
                    
        predictions.extend(y_pred)
        actuals.extend(y_test)
    
    mape_rt = mean_absolute_percentage_error(actuals, predictions)
    
    return predictions, actuals, test_indices, model, mape_rt


# Perform rolling window validation
len(DF)
train_size = 340
test_size = 50


train_size = 400
test_size = 50

predictions, actuals, test_indices, model, mape_rt =\
rolling_window_validation(new_DF, 'sales', 3, train_size,\
  test_size, features, model_type='linear_regression', transform_fun='logistic')

   
###############

data=pd.read_csv(r"C:\lsg\重要项目\tmseries\Electric_Production.csv",parse_dates=['DATE'],index_col='DATE')
monthly_data = data.IPG2211A2N.resample('M').mean().reset_index()
max_y = monthly_data['IPG2211A2N'].max()

monthly_data['IPG2211A2N'] = 1/(1 + np.exp(-1*monthly_data['IPG2211A2N']/max_y))

# Splitting the data into training and test sets
train_data = np.log(monthly_data['IPG2211A2N'][:-3])
test_data = np.log(monthly_data['IPG2211A2N'][-3:])

max_y = new_DF['sales'].max()
train_data = 1/(1 + np.exp(-new_DF['sales'][:-5]/max_y))
test_data = 1/(1 + np.exp(-new_DF['sales'][-5:]/max_y))

model_sarima = SARIMAX(train_data, order=(1, 1, 0), seasonal_order=(1, 1, 1, 12))
model_sarima_fit = model_sarima.fit()

# Forecast the last three data points
forecast_sarima = model_sarima_fit.forecast(steps=5)

test_data, forecast_sarima = test_data**2, forecast_sarima**2
test_data, forecast_sarima = np.exp(test_data), np.exp(forecast_sarima)
test_data, forecast_sarima = -np.log((1/test_data)- 1)*max_y, -np.log((1/forecast_sarima)-1)*max_y

mape_sarima = mean_absolute_percentage_error(test_data, forecast_sarima)

##################    

# Calculate validation metrics: mape_rt
Electric_Production_tm.csv:
trasnform: none:    
lightgbm: 0.031 
linear_regression: 0.046
xgboost: 0.038
random_forest: 0.034
SARIMAX: 0.442

Electric_Production_tm.csv:
trasnform: sqrt: 
lightgbm: 0.031  
linear_regression: 0.052
xgboost: 0.036
random_forest: 0.034
SARIMAX: 0.043

Electric_Production_tm.csv:
trasnform: log: 
lightgbm: 0.031  
linear_regression: 0.068
xgboost: 0.032
random_forest: 0.033
SARIMAX: 0.042

Electric_Production_tm.csv:
trasnform: logistic: 
lightgbm: 0.031  
linear_regression: 0.057
xgboost: 0.036
random_forest: 0.034
SARIMAX: 0.045

######################################
simulation.csv:
trasnform: none:    
lightgbm: 0.092 
linear_regression: 0.082
xgboost: 0.086
random_forest: 0.081
SARIMAX: 0.159

simulation.csv:
trasnform: sqrt:    
lightgbm: 0.094 
linear_regression: 0.081
xgboost: 0.083
random_forest: 0.080
SARIMAX: 0.163

simulation.csv:
trasnform: log:    
lightgbm: 0.092 
linear_regression: 0.081
xgboost: 0.085
random_forest: 0.080
SARIMAX: 0.166

simulation.csv:
trasnform: logistic:    
lightgbm: 0.095 
linear_regression: 0.081
xgboost: 0.085
random_forest: 0.081
SARIMAX: 0.160

###forecasting######
def forecast_next_1_day(model, DF, features):
    # Ensure the dataframe is sorted by date
    DF = DF.sort_values('date')

    # Create the next day's date
    next_date = DF['date'].max() + pd.Timedelta(days=1)

    # Create a new row for the next day's features
    new_row = pd.DataFrame(index=[0])

    # Fill the new row with feature values
    new_row['date'] = next_date
    new_row['t'] = (next_date - DF['date'].min()).days
    new_row['t_lg'] = np.log(new_row['t'] + 1)

    # Calculate lag features
    for lag in range(1, 4):  # Assuming lag_n=3 from the provided create_date function
        new_row[f'lag{lag}'] = DF['sales'].shift(lag).iloc[-1]

    # Calculate rolling statistics
    new_row['rolling_mean_7'] = DF['sales'].rolling(window=7).mean().iloc[-1]
    new_row['rolling_std_7'] = DF['sales'].rolling(window=7).std().iloc[-1]
    new_row['rolling_mean_30'] = DF['sales'].rolling(window=30).mean().iloc[-1]
    new_row['rolling_std_30'] = DF['sales'].rolling(window=30).std().iloc[-1]

    # Extract day of the week and month for dummy variables
    new_row['weekday'] = next_date.day_name()
    new_row['month'] = next_date.month_name()

    dummies_wkd = pd.get_dummies(new_row['weekday'], prefix='is')
    new_row = pd.concat([new_row, dummies_wkd], axis=1)

    dummies_mon = pd.get_dummies(new_row['month'], prefix='is')
    new_row = pd.concat([new_row, dummies_mon], axis=1)

    holidays = cal.holidays(start=DF['date'].min(), end=DF['date'].max())
    new_row['is_holiday'] = next_date in holidays

    # Fill missing dummy variables with 0 if they do not exist
    missing_columns = set(features) - set(new_row.columns)
    for col in missing_columns:
        new_row[col] = 0

    # Ensure the new row contains only the required features
    new_row = new_row[features + ['date']]

    # Predict the sales for the next day
    sales_prediction = model.predict(new_row[features])[0]

    return sales_prediction, new_row

def forecast_next_7_days(model, DF, features):
    predictions = []
    for _ in range(7):
        sales_prediction, new_row = forecast_next_1_day(model, DF, features)
        predictions.append(sales_prediction)

        # Append the prediction to the DataFrame to update lags and rolling stats for the next iteration
        new_row['sales'] = sales_prediction
        DF = pd.concat([DF, new_row[['date', 'sales'] + features]], ignore_index=True)

    return predictions, DF

# Predict the next 7 days sales and update the DataFrame
DF = new_DF.copy()
seven_day_preds, updated_DF = forecast_next_7_days(model, DF, features)

# Add the 'pred' column for testing data predictions
new_DF['pred'] = np.nan
new_DF.loc[test_indices, 'pred'] = predictions

# Add the new forecasted sales data to new_DF
forecast_dates = pd.date_range(start=new_DF['date'].max() + pd.Timedelta(days=1), periods=7)
forecast_df = pd.DataFrame({'date': forecast_dates, 'pred': seven_day_preds, 'sales': [np.nan] * 7})
new_DF = pd.concat([new_DF, forecast_df], ignore_index=True)

# Display the updated DataFrame with predictions
print(new_DF.tail(15))

new_DF[new_DF.pred.isnull()==False][['date', 'sales', 'pred']].head(100)
new_DF[new_DF.pred.isnull()==False][['date', 'sales', 'pred']].tail(20)

#############
# Plot 1: Time series dots plot with training and testing data
# Plot 1: Time series dots plot with training and testing data
plt.figure(figsize=(14, 7))
plt.scatter(new_DF['date'][:train_size], new_DF['sales'][:train_size], color='blue', label='Training Data', alpha=0.6)
plt.scatter(new_DF['date'][train_size:train_size + test_size], new_DF['sales'][train_size:train_size + test_size], color='blue', label='Actual Testing Data', alpha=0.6)
plt.scatter(new_DF['date'][train_size:train_size + test_size], new_DF['pred'][train_size:train_size + test_size], color='red', label='Predicted Testing Data', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Training and Testing Data')
plt.legend()
plt.show()


# Plot 2: Last 25 points in current data and next 7 days forecasted sales
plt.figure(figsize=(14, 7))
last_25_data = new_DF[-(25 + 7):-7]
forecast_data = new_DF[-7:]
forecast_data.shape
last_25_data.shape

plt.scatter(last_25_data['date'], last_25_data['sales'], color='blue', label='Last 25 Days', alpha=0.6)
plt.plot(forecast_data['date'], forecast_data['pred'], color='red', linestyle='-', linewidth=1, marker='o', label='Next 7 Days Forecast', alpha=0.6)

plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Last 25 Days and Next 7 Days Forecast')
plt.legend()
plt.show()












