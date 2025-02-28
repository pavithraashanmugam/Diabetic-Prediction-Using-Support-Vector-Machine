WORK FLOW   

Diabetes data --> dada pre-processing --> train test split --> support vector machine classifier    

new data --> trained support vector machine classifier --> outcome

1. Importing the dependencies

# numpy arrays
import numpy as np
# pandas data frame
import pandas as pd
# standardizing to common range
from sklearn.preprocessing import StandardScaler
# train & test data
from sklearn.model_selection import train_test_split
# support vector machine
from sklearn import svm
# accuracy
from sklearn.metrics import accuracy_score


2. Data collection & analysis

# loading the dataset to pandas data frame
diabetes_data = pd.read_csv('/content/drive/MyDrive/Machine Learning Dataset/diabetes.csv')

diabetes_data.head()

diabetes_data.shape

# statistical measures of the data
diabetes_data.describe()

diabetes_data['Outcome'].value_counts()

0 --> Non diabetic   
1 --> diabetic

diabetes_data.groupby('Outcome').mean()


3. Seperating data & labels

X = diabetes_data.drop(columns = 'Outcome', axis =1)
Y = diabetes_data['Outcome']

print(X)

print(Y)

"""4. Standardize the data"""

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)
# scaler.fitTransform (X) can also be done together

print(standardized_data)

# feeding standardized_data to X
X = standardized_data
Y = diabetes_data['Outcome']

print(X)

print(Y)


5. Train Test Split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state = 2)

print(X.shape,x_train.shape,x_test.shape)


6. Train the model

# SVC - support Vector Classifier
# kernel --> linear model
classifier = svm.SVC(kernel = 'linear')

# training the SVC
classifier.fit(x_train, y_train)


7. Evaluate the model

# finding accuracy score for the training data

# pedicted lable of x_train is stored in x_train_prediction
x_train_prediction = classifier.predict(x_train)

# the x_train_prediction (predicted value) is compared with y_train (label)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

# finding accuracy score for the test data

# pedicted lable of x_test is stored in x_test_prediction
x_test_prediction = classifier.predict(x_test)

# the x_test_prediction (predicted value) is compared with y_test (label)
testing_data_accuracy = accuracy_score(x_test_prediction,y_test)

print('accuracy of training data :',training_data_accuracy)

print('accuracy of testing data :',testing_data_accuracy)


8. Making the predictive system

input_data = (4,110,92,0,0,37.6,0.191,30)
# change the input_data (list) to numpy array
input_as_numpy = np.asarray(input_data)
# The input should be reshaped for 1 instance to match the model's expected format, as it was trained on multiple data instances.
input_reshaped = input_as_numpy.reshape(1,-1)
# standardize the input data
std_data = scaler.transform(input_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if(prediction[0] == 0):
  print('The person is Non-Diabetic')
else:
  print('The person is Diabetic')

