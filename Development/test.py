# Selection Statement On Dataset
import pandas as pd
import numpy as np
from fold_cross_validation import fold_cross_validation
from random_forest_apply import random_forest
from k_near_neighbor_apply import k_near_neighbor
from support_vector_machine_apply import support_vector_machine

# DATASET SELECTION -------------------------------------------------------------------------------------------------------------------------
selection_dataset = 'Burnout'

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

elif selection_dataset == 'Satisfaction':
    # Data Reading
    data = pd.read_csv("../Datasets/employee_satisfaction.csv")

    # Data Splitting
    data_input = data.drop(columns=['Name', 'Feedback Score', 'Joining Date', 'Satisfaction Rate (%)'])
    data_target = data['Satisfaction Rate (%)']
    x_train, y_train, x_test, y_test = fold_cross_validation (data_input, data_target)

    # Dataset Type
    reg_or_class = 'Regression'
    user_input_array = np.array([])

elif selection_dataset == 'Burnout':
    # Data Reading
    data_train = pd.read_csv("../Datasets/employe_burnout/train.csv")
    data_test = pd.read_csv("../Datasets/employe_burnout/test.csv")
    
    # Data Cleaning for Data Train
    for column in ['Resource Allocation', 'Mental Fatigue Score']:
        average_value = data_train[column].mean()
        data_train[column].fillna(average_value, inplace=True)

    # Data Cleaning for Data Test
    for column in ['Resource Allocation', 'Mental Fatigue Score']:
        average_value = data_test[column].mean()
        data_test[column].fillna(average_value, inplace=True)

    # Assuming you have categorical columns that need encoding
    categorical_columns = ['Gender', 'Company Type', 'WFH Setup Available']
    data_train = pd.get_dummies(data_train, columns=categorical_columns)
    data_test = pd.get_dummies(data_test, columns=categorical_columns)

    #df = df.drop(index=range(2000, len(df)))
    # Drop row-2001 until the end, Learning process to slow
    data_train = data_train.drop(index=range(2000, len(data_train)))
    data_test = data_test.drop(index=range(2000, len(data_test)))

    # Data Splitting
    x_train = data_train.drop(columns=['Employee ID', 'Date of Joining', 'Mental Fatigue Score','Burn Rate'])
    y_train = data_train['Mental Fatigue Score']
    x_test = data_test.drop(columns=['Employee ID', 'Date of Joining', 'Mental Fatigue Score'])
    y_test = data_test['Mental Fatigue Score']

    # Dataset Type
    reg_or_class = 'Regression'
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
        print("test result (n=1): ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 1, user_input_array)
        print("test result (n_estimat=1 ): ", test_result, "(", test_result_type, ")")
        est_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 5, user_input_array)
        print("test result (n_estimat=5 ): ", test_result, "(", test_result_type, ")")
        est_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 10, user_input_array)
        print("test result (n_estimat=10): ", test_result, "(", test_result_type, ")")
        est_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 15, user_input_array)
        print("test result (n_estimat=15): ", test_result, "(", test_result_type, ")")
    
elif selection_algorithm == 'KNN' :

    if np.any(user_input_array):
        test_result_type, test_result, user_pred = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 100, user_input_array)
        print("test result : ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        test_result_type, test_result = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 100, user_input_array)
        print("test result (n_neighbour=100): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 200, user_input_array)
        print("test result (n_neighbour=200): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 300, user_input_array)
        print("test result (n_neighbour=300): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 400, user_input_array)
        print("test result (n_neighbour=400): ", test_result, "(", test_result_type, ")")

elif selection_algorithm == 'SVM' :

    if np.any(user_input_array):
        test_result_type, test_result, user_pred = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'rbf', user_input_array)
        print("test result : ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'linear', user_input_array)
        print("test result (linear ): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'linear', user_input_array)
        print("test result (  rbf  ): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'linear', user_input_array)
        print("test result (sigmoid): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'linear', user_input_array)
        print("test result (  poly ): ", test_result, "(", test_result_type, ")")

else :
    print('None')
