# Selection Statement On Dataset
import pandas as pd
import numpy as np
from fold_cross_validation import fold_cross_validation
from random_forest_apply import random_forest
from k_near_neighbor_apply import k_near_neighbor
from support_vector_machine_apply import support_vector_machine

# DATASET SELECTION -------------------------------------------------------------------------------------------------------------------------
selection_dataset = 'Turnover'

if selection_dataset == 'Absenteeism':
    # Data Reading
    data = pd.read_csv("../Datasets/Absenteeism_at_work.csv")
    
    # Data Splitting
    data_input = data.drop(columns=['Work load Average/day ', 'Hit target', 'Absenteeism time in hours'])
    data_target = data['Absenteeism time in hours']
    x_train, y_train, x_test, y_test = fold_cross_validation (data_input, data_target)
    
    # Dataset Type
    reg_or_class = 'Regression'
    
    # streamlit
    #user_input_array = np.array([[11, 26, 7, 3, 1, 289, 36, 13, 33, 0, 1, 2, 1, 0, 1, 90, 172, 30], [11, 26, 7, 3, 1, 289, 36, 13, 33, 0, 1, 2, 1, 0, 1, 90, 172, 30]])
    user_input_array = np.array([])
    
elif selection_dataset == 'Turnover':
     # Data Reading
    data = pd.read_csv("../Datasets/turnover.csv", encoding='latin1')
    
    # Assuming you have categorical columns that need encoding
    categorical_columns = ['gender', 'industry', 'profession', 'traffic', 'coach', 'head_gender', 'greywage', 'way']
    data = pd.get_dummies(data, columns=categorical_columns)
    
    # Data Splitting
    data_input = data.drop(columns=['novator', 'event'])
    data_target = data['event']
    x_train, y_train, x_test, y_test = fold_cross_validation (data_input, data_target)
    
    # Dataset Type
    reg_or_class = 'Classification'
    user_input_array = np.array([])
    
else:
    print("None")


# ALGORITHM SELECTION -------------------------------------------------------------------------------------------------------------------------
    
selection_algorithm = 'SVM'

if selection_algorithm == 'Random Forest' :

    if np.any(user_input_array):
        test_result_type, test_result, user_pred = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 1, user_input_array)
        print("test result : ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 1, user_input_array)
        print("test_result : ", test_result, "(", test_result_type, ")")
    
elif selection_algorithm == 'KNN' :

    if np.any(user_input_array):
        test_result_type, test_result, user_pred = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 500, user_input_array)
        print("test result : ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        test_result_type, test_result = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 500, user_input_array)
        print("test_result : ", test_result, "(", test_result_type, ")")

elif selection_algorithm == 'SVM' :

    if np.any(user_input_array):
        test_result_type, test_result, user_pred = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'rbf', user_input_array)
        print("test result : ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'sigmoid', user_input_array)
        print("test_result : ", test_result, "(", test_result_type, ")")

else :
    print('None')
