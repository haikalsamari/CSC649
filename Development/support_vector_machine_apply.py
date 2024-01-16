# Method Definition : Random Forest Model Training
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

def random_forest (x_train, x_test, y_train, y_test, reg_or_class, n_estimator, user_input_array):
    
    n = n_estimator
    
    # Identfication Whether The Dataset is Regressor or Class
    if reg_or_class == 'Regression' :
        
        # Develop the model
        svm = SVC(kernel=n)
        
        # Train model
        svm.fit(x_train, y_train)
        
        # Test model (prediction)
        y_pred = svm.predict(x_test)
        
        # Test model (evaluation)
        test_result = mean_squared_error(y_test, y_pred)
        test_result_type = "mse"

        
    elif reg_or_class == 'Classification' :
        
        # Develop the model
        svm = SVR(kernel=n)
        
        # Train model
        svm.fit(x_train, y_train)
        
        # Test model (prediction)
        y_pred = svm.predict(x_test)
        
        # Test model (evaluation)
        test_result = accuracy_score(y_test, y_pred)
        test_result_type = "accuracy"
        
    # if there's input from user
    if user_input_array.size > 0:
        user_pred = svm.predict(user_input_array)
        
        return test_result_type, test_result, user_pred
    else: 
        return test_result_type, test_result