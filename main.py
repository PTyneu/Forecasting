import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

df=pd.read_csv('/datasets/taxi.csv', index_col=[0], parse_dates=[0])
df.sort_index(inplace=True)

df = df.resample('1H').sum()
df.dropna()
df.info()

#Оценка сезонности
decomposed = seasonal_decompose(df)

plt.figure(figsize=(6, 8))
plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.xlim('2018-08-01', '2018-08-30')
plt.ylim(-100, 100)
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')
plt.tight_layout()

df['rolling_mean'] = df.rolling(100).mean()
df.plot(figsize=[20, 10])
df.describe()

#Генерация фичей (скользащее среднее и смещения)
def make_features(data, max_lag, rolling_mean_size):
    data['dayofweek'] = data.index.dayofweek
    data['hour'] = data.index.hour
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)

    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()
make_features(df, 2, 24)

df_train, df_test = train_test_split(df, test_size=0.1, shuffle=False, random_state=12345)
train_features = df_train.drop('num_orders', axis=1)
train_features=train_features.fillna(value=None, method="bfill")
train_target = df_train['num_orders']
test_features=df_test.drop('num_orders', axis=1)
test_features=test_features.fillna(value=None, method="bfill")
test_target = df_test['num_orders']

def fitrm (model):
    model.fit(train_features, train_target)
    test_predict=pd.Series(model.predict(train_features))
    mse=mean_squared_error(train_target, test_predict)
    rmse=mse**0.5
    print("Точность модели", model.score(test_features, test_target))
    print('RMSE модели:', rmse)

#Линейка без параметров дабы просто было с чем сравнивать
model_lr = LinearRegression()
fitrm(model_lr)

#Дерево решений

model = DecisionTreeRegressor(random_state=12345)
best_model = None
best_result = 0
best_leaf = 0
best_depth = 0
best_split = 2
for split in range(3, 11):
    for leaf in range(1, 11):
        for depth in range (1, 15):
            model = DecisionTreeRegressor(random_state=12345, max_depth=depth, min_samples_leaf=leaf, min_samples_split= split)
            model.fit(train_features, train_target)
            result = model.score(test_features, test_target)
            if result > best_result:
                best_model = model
                best_result = result
                best_leaf = leaf
                best_depth = depth
                best_split=split
print("Точность наилучшей модели:", best_result,
      "минимум значений в листе?:", best_leaf,
      "Максимальная глубина:", depth, "лучшее разделение:", best_split)

#Cat
model_cat=CatBoostRegressor(iterations=100,
                          learning_rate=0.25,
                          depth=5, random_seed=12345, verbose=0)
fitrm(model_cat)


model_gbm = lgb.LGBMRegressor(n_estimators=150, max_depth=4,
                              boosting_type = 'gbdt', random_state=12345)
fitrm(model_gbm)

#Случайный лес

best_model = None
best_result = 0
best_est = 0
best_depth = 0
for est in range(1, 101, 10):
    for depth in range (1, 15):
        model = RandomForestRegressor(random_state=12345, n_estimators=est, max_depth=depth)
        model.fit(train_features, train_target)
        result = model.score(test_features, test_target)
        if result > best_result:
            best_model = model
            best_result = result
            best_est = est
            best_depth = depth

print("Точность наилучшей модели:", best_result,
      "Количество деревьев:", best_est, "Максимальная глубина:", depth)

import seaborn as sns
def fittest (model):
    model.fit(train_features, train_target)
    preds_test=model.predict(test_features)
    mse=mean_squared_error(test_target, preds_test)
    rmse=mse**0.5
    print("RMSE модели на тестовой выборке", rmse)
    predictions = pd.Series(preds_test)
    predictions.index = test_target.index

    fig2, ax = plt.subplots(figsize=(8, 3))
    ax = sns.lineplot(data=test_target, label='Реальные')
    sns.lineplot(data=predictions, color='red', label='Предсказанные')
    plt.xticks(rotation=90)
    plt.xlim('2018-08-24', '2018-08-27')
    plt.title('Результаты модели, 2 дня')
    plt.xlabel('Дата и время')
    plt.ylabel('Число заказов');

print('Catboost')
fittest(model_cat)

print('LightGBM')
fittest(model_gbm)

print('Random forest')
model_rf=model = RandomForestRegressor(random_state=12345, n_estimators=81, max_depth=14)
fittest(model_rf)

dumb=np.ones(test_target.shape) * test_target.mean()
mse=mean_squared_error(test_target, dumb)
rmse=mse**0.5
print("RMSE вектора средних на тестовой выборке", rmse)
