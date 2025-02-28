### Diabetes Prediction using Support Vector Machine (SVM)

**Project Overview**:  
This project implements a diabetes prediction system using the **Support Vector Machine (SVM)** algorithm. The model classifies whether a person is diabetic or non-diabetic based on various medical features.

### Steps:

1. **Import Dependencies**:  
   Import necessary libraries such as `numpy`, `pandas`, `StandardScaler`, `accuracy_score`, and `train_test_split` from `sklearn`.

2. **Data Collection & Analysis**:  
   - Load the dataset by converting the CSV file into a pandas DataFrame.  
   - Use the `describe()` method to obtain statistical summaries of the dataset.  
   - Utilize `value_counts()` to understand the distribution of the target variable (Diabetic/Non-Diabetic).

3. **Separate Features and Labels**:  
   - Split the dataset into features (input data) and labels (target variable).

4. **Standardize the Data**:  
   - Apply `StandardScaler` to standardize the features, ensuring the model works optimally.

5. **Train-Test Split**:  
   - Split the dataset into training and testing sets using `train_test_split` to validate the model's performance.

6. **Train the Model**:  
   - Train the **Support Vector Machine (SVM)** model on the training data.

7. **Model Evaluation**:  
   - Evaluate the model's performance using `accuracy_score` to assess its prediction accuracy on the test data.

8. **Prediction**:  
   - Build a predictive model that takes new input data. Strandardizes the input data and predicts whether a person is diabetic or not.

This project demonstrates how to apply machine learning techniques, specifically **Support Vector Machine**, to predict diabetes based on medical data.
