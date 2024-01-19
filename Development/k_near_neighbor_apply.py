# Method Definition : Random Forest Model Training
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

def k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, n_estimator, user_input_array):
    
    test_result = 0
    test_result_type = ""
    n = n_estimator
    
    # Identfication Whether The Dataset is Regressor or Class
    if reg_or_class == 'Regression' :
        
        # Develop the model
        rf = KNeighborsRegressor(n_neighbors=n)
        
        # Train model
        rf.fit(x_train, y_train)
        
        # Test model (prediction)
        y_pred = rf.predict(x_test)
        
        # Test model (evaluation)
        test_result = mean_squared_error(y_test, y_pred)
        test_result_type = "mse"

        
    elif reg_or_class == 'Classification' :
        
        # Develop the model
        rf = KNeighborsClassifier(n_neighbors=n)
        
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