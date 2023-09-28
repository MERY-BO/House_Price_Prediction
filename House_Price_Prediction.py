import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('housing.csv')

df.head(10)

df.info()
df.columns
df.dropna(inplace = True)


df.hist(figsize = (15,8))

plt.figure(figsize = (15,8))
sns.heatmap(df.corr(),annot = True)


df['total_bedrooms'] = np.log(df['total_bedrooms'] + 1)
df['total_rooms']    = np.log(df['total_rooms'] + 1)
df['population'] = np.log(df['population'] + 1)
df['households'] = np.log(df['households'] + 1)

df['ocean_proximity'].value_counts()
df = df.join(pd.get_dummies(df['ocean_proximity'])).drop('ocean_proximity',axis = 1)

sns.scatterplot(x = 'longitude',y  = 'latitude', data = df, hue = 'median_house_value' )

x = df.drop('median_house_value',axis = 1)
y = df['median_house_value']

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2)

model = LinearRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)


mod2 = RandomForestRegressor()
mod2.fit(x_train,y_train)
mod2.score(x_test,y_test)

param_grid = {
    'n_estimators' : [3,5,10],
    'max_features' : [2,4,6,8,12,20]
}

grid_search = GridSearchCV(mod2,param_grid,cv = 5)
grid_search.fit(x_train,y_train)

best = grid_search.best_estimator_

best.score(x_test,y_test)