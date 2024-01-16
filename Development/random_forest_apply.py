# Method Definition : Random Forest Model Training
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

def random_forest (x_train, x_test, y_train, y_test, reg_or_class, n_estimator, user_input_array):
    
    n = n_estimator
    
    # Identfication Whether The Dataset is Regressor or Class
    if reg_or_class == 'Regression' :
        
        # Develop the model
        rf = RandomForestRegressor(n_estimators=n)
        
        # Train model
        rf.fit(x_train, y_train)
        
        # Test model (prediction)
        y_pred = rf.predict(x_test)
        
        # Test model (evaluation)
        test_result = mean_squared_error(y_test, y_pred)
        test_result_type = "mse"

        
    elif reg_or_class == 'Classification' :
        
        # Develop the model
        rf = RandomForestClassifier(n_estimators=n)
        
        # Train model
        rf.fit(x_train, y_train)
        
        # Test model (prediction)
        y_pred = rf.predict(x_test)
        
        # Test model (evaluation)
        test_result = accuracy_score(y_test, y_pred)
        test_result_type = "accuracy"
        
    # if there's input from user
    if user_input_array.size > 0:
        user_pred = rf.predict(user_input_array)
        
        return test_result_type, test_result, user_pred
    else: 
        return test_result_type, test_result