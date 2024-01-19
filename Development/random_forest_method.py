from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

def RandomForest(user_input_array):
    if np.any(user_input_array):
        test_result_type, test_result, user_pred = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 1, user_input_array)
        print("test result (n=1): ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 1, user_input_array)
        print("test result (n_estimate=1 ): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 5, user_input_array)
        print("test result (n_estimate=5 ): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 10, user_input_array)
        print("test result (n_estimate=10): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 15, user_input_array)
        print("test result (n_estimate=15): ", test_result, "(", test_result_type, ")")