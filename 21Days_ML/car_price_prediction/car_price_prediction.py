import numpy as np
import pandas as pd
import pickle #to save the model
### scikit learn libraries
from sklearn.preprocessing import LabelEncoder #module to encode non numerical values
from sklearn.model_selection import train_test_split #module to split data to training data & test data
from sklearn.ensemble import RandomForestRegressor #module to train using random forest


df = pd.read_csv('car_data.csv')

### filtering dependent & independent values from dataframe
x = df.iloc[:, [1,3,4,6]].values #indpendent values : year, km driven, fuel, transmission
y = df.iloc[:, 2].values #dependent value : price

### encodeing non numerical values using LabelEncoder() class
fuel_lbl_enc = LabelEncoder() #object for label encoder class
trans_lbl_enc = LabelEncoder() #object for label encoder class
x[: ,2] = fuel_lbl_enc.fit_transform(x[:, 2]) #label encoding fuel data
x[:,3] = trans_lbl_enc.fit_transform(x[:, 3]) #label encoding transmission data

###spliting dataset testing & training models using train_test_split() function
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# x & y are datasets to split
# test_size is the ratio to split 0.2 means 80% training data & 20% testing data
# random_state is set to 0 to ensure test_size is not set randomly

### training model using random forest generator
model = RandomForestRegressor(n_estimators = 28, random_state = 0) #n_estimate is the number of decision trees
model.fit(x_train, y_train) #training the model with data set
#model.score() is used to test the accuracy of the model
accuracy = model.score(x_test, y_test) * 100
model.fit(x_train, y_train) #training the model with data set
print('Accuracy of the model:', accuracy, '%')

#testing
new_data = []
new_data.append(int(input('Year: ')))
new_data.append(int(input('Km Driven: ')))
new_data.append(input('Fuel: ').capitalize())
new_data.append(input('Transmission: ').capitalize())
new_data[2] = fuel_lbl_enc.transform([new_data[2]])[0]
new_data[3] = trans_lbl_enc.transform([new_data[3]])[0]
print(chr(8377), round(model.predict([new_data])[0], 2))

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(fuel_lbl_enc, open('fuel_lbl_enc', 'wb'))
pickle.dump(trans_lbl_enc, open('trans_lbl_enc', 'wb'))
