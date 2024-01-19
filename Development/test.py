# Selection Statement On Dataset
import pandas as pd
import numpy as np
import streamlit as st
from array import array
from fold_cross_validation import fold_cross_validation
from random_forest_apply import random_forest
from k_near_neighbor_apply import k_near_neighbor
from support_vector_machine_apply import support_vector_machine

# Declaration and Initialization of Variables
user_input_array = np.array([])
x_train, x_test, y_train, y_test = array('i', [0, 0, 0, 0])
reg_or_class = " "
test_result_type = " "
test_result = 0

# Title of the Project
st.title("PREDICTION & ANALYSIS ON EMPLOYEE BEHAVIOR AND PRODUCTIVITY WITHIN IT RELATED COMPANY")
st.write("Workforce analytics is a vital tool in the IT sector which offering insights into individual and team performance, "
         "project efficiency and employee satisfaction. By analyzing metrics such as time to hire and retention rate, it shapes organizational strategies and maximizes human capital potential. "
         "Meanwhile, machine learning is a subset of artificial intelligence that can enhances efficiency, automates repetitive tasks and provides accurate data analysis. "
         "Leveraging historical data patterns, machine learning aids in predicting employee turnover that can allow companies to address productivity challenges and optimize operations.")

# Project Objectives
st.header("Project Objectives")
st.write("1. To identify machine learning techniques or algorithms that are suitable for predicting employees' behavior and productivity within IT companies.")
st.write("2. To design and develop a prototype model that has the potential to predict employees' behavior and productivity within IT related companies.")
st.write("3. To leverage multiple machine learning algorithms and techniques into a predicting model that can ease any workforce analytic task by the HR department within IT companies.")
st.write("4. To test and evaluate the accuracy of the prototype model to predict employees' behavior and productivity within IT companies.")

# Project Scopes
st.header("Project Scopes")
st.write("1. This project will target the employees within IT companies to predict their behavior and productivity while working.")
st.write("2. This project will implement three suitable machine learning techniques which are k-Nearest Neighbors (kNN), Support Vector Machine (SVM) and Random Forest (RF) in the project.")
st.write("3. This project will develop a predictive model that incorporates various machine learning algorithms to streamline and enhance workforce analytics tasks for the HR department in IT companies.")

# Title on Sidebar
st.sidebar.title("Prediction & Analysis On Employee Behavior And Productivity Within IT Related Company")

# Dropdown List for Dataset Selection
selection_dataset = st.sidebar.selectbox("Select Dataset", options = ["Select Dataset", "Burnout", "Absenteeism", "Satisfaction", "Turnover"])

# Checkbox for Algorithm Selection
selection_algorithm = st.sidebar.multiselect("Select Algorithm", ["Random Forest", "KNN", "SVM"])


# --------------------------------------------------- METHOD OF ALGORITHM SELECTION ---------------------------------------------------
def RandomForest(user_input_array):
    if np.any(user_input_array):
        test_result_type, test_result, user_pred = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 1, user_input_array)
        print("test result (n=1): ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 1, user_input_array)
        st.subheader("N_Estimation = 1")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (n_estimate=1 ): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 5, user_input_array)
        st.subheader("N_Estimation = 5")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (n_estimate=5 ): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 10, user_input_array)
        st.subheader("N_Estimation = 10")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (n_estimate=10): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 15, user_input_array)
        st.subheader("N_Estimation = 15")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (n_estimate=15): ", test_result, "(", test_result_type, ")")

def KNearestNeighbors(user_input_array):
    if np.any(user_input_array):
        test_result_type, test_result, user_pred = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 100, user_input_array)
        print("test result : ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        test_result_type, test_result = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 100, user_input_array)
        st.subheader("N_Neighbor = 100")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (n_neighbor = 100): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 200, user_input_array)
        st.subheader("N_Neighbor = 200")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (n_neighbor = 200): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 300, user_input_array)
        st.subheader("N_Neighbor = 300")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (n_neighbor = 300): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 400, user_input_array)
        st.subheader("N_Neighbor = 400")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (n_neighbor = 400): ", test_result, "(", test_result_type, ")")

def SupportVectorMachine(user_input_array):
    if np.any(user_input_array):
        test_result_type, test_result, user_pred = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'rbf', user_input_array)
        print("test result : ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'linear', user_input_array)
        st.subheader("Kernel Linear")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (linear): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'rbf', user_input_array)
        st.subheader("Kernel RBF")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (rbf): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'sigmoid', user_input_array)
        st.subheader("Kernel Sigmoid")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (sigmoid): ", test_result, "(", test_result_type, ")")
        test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'poly', user_input_array)
        st.subheader("Kernel Poly")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        print("test result (  poly ): ", test_result, "(", test_result_type, ")")


# --------------------------------------------------- DATASET SELECTION ---------------------------------------------------
# Burnout
if selection_dataset == 'Burnout':

    # Title of Dataset
    st.header("Employee Burnout Dataset")
    st.write("Employee burnout at work is a pervasive issue characterized by exhaustion, reduced productivity, and emotional detachment. "
             "It stems from chronic workplace stress, adversely impacting both individual well-being and organizational performance.")

    # Data Reading
    data_train = pd.read_csv("../Datasets/employe_burnout/train.csv")
    data_test = pd.read_csv("../Datasets/employe_burnout/test.csv")

    # Display Both Datasets
    st.subheader("Full Dataset of Employee Burnout")
    st.write("This dataset mainly focused on the training data of employee burnout.")
    data_train
    st.write("While, this dataset mainly focused on the testing data of employee burnout.")
    data_test
    
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

    # Drop row-2001 until the end, Learning process to slow
    data_train = data_train.drop(index=range(2000, len(data_train)))
    data_test = data_test.drop(index=range(2000, len(data_test)))

    # Data Splitting
    x_train = data_train.drop(columns=['Employee ID', 'Date of Joining', 'Mental Fatigue Score','Burn Rate'])
    y_train = data_train['Mental Fatigue Score']
    x_test = data_test.drop(columns=['Employee ID', 'Date of Joining', 'Mental Fatigue Score'])
    y_test = data_test['Mental Fatigue Score']

    # Display Corresponding Training Data
    st.subheader("Input Training Data")
    st.write("As provided below, this dataset is specifically used for the input of training data.")
    x_train
    st.subheader("Target Training Data")
    st.write("In the meanwhile, this dataset is specifically used for the target of training data.")
    y_train

    # Display Corresponding Testing Data
    st.subheader("Input Testing Data")
    st.write("As provided below, this dataset is specifically used for the input of testing data.")
    x_test
    st.subheader("Target Testing Data")
    st.write("In the meanwhile, this dataset is specifically used for the target of testing data.")
    y_test

    # Dataset Type
    reg_or_class = 'Regression'
    user_input_array = np.array([])



    # Multi-Selection of Algorithms
    if "Random Forest" in selection_algorithm:
        st.header("Test Result of Random Forest for Employee Burnout Dataset")
        RandomForest(user_input_array)
    if "KNN" in selection_algorithm:
        st.header("Test Result of KNN for Employee Burnout Dataset")
        KNearestNeighbors(user_input_array)
    if "SVM" in selection_algorithm:
        st.header("Test Result of SVM for Employee Burnout Dataset")
        SupportVectorMachine(user_input_array)
    if not selection_algorithm:
        print("Please Select At Least One Algorithm")

# Absenteeism
elif selection_dataset == 'Absenteeism':
    
    # Title of Dataset
    st.header("Employee Absenteeism At Work Dataset")
    st.write("Employee absenteeism refers to the frequent or prolonged absence of employees from work. "
             "It can be caused by various factors such as illness, personal reasons, or dissatisfaction."
             "High absenteeism rates may impact productivity and require effective management strategies.")
    
    # Data Reading
    data = pd.read_csv("../Datasets/Absenteeism_at_work.csv")

    # Display Full Datasets
    st.subheader("Full Dataset of Employee Absenteeism At Work")
    st.write("This dataset contains full dataset of employee absenteeism at work.")
    data
    
    # Data Splitting
    data_input = data.drop(columns=['Work load Average/day ', 'Hit target', 'Absenteeism time in hours'])
    data_target = data['Absenteeism time in hours']
    x_train, y_train, x_test, y_test = fold_cross_validation (data_input, data_target)
    
    # Display Both Input and Target Data
    st.subheader("Input Data of Employee Absenteeism At Work")
    st.write("It contains input data of employee absenteeism at work.")
    data_input
    st.subheader("Target Data of Employee Absenteeism At Work")
    st.write("In addition, the target data of employee absenteeism at work is included as well.")
    data_target
    
    # Display Corresponding Training Data
    st.subheader("Input Training Data of Employee Absenteeism At Work")
    st.write("Input training data of employee absenteeism at work is provided below.")
    x_train
    st.subheader("Target Training Data of Employee Absenteeism At Work")
    st.write("Furthermore, target training data of employee absenteeism at work is provided as well.")
    y_train

    # Display Corresponding Testing Data
    st.subheader("Input Testing Data of Employee Absenteeism At Work")
    st.write("Input testing data of employee absenteeism at work is provided below.")
    x_test
    st.subheader("Target Testing Data of Employee Absenteeism At Work")
    st.write("Furthermore, target testing data of employee absenteeism at work is provided as well.")
    y_test

    # Dataset Type
    reg_or_class = 'Regression'
    
    # user_input_array = np.array([[11, 26, 7, 3, 1, 289, 36, 13, 33, 0, 1, 2, 1, 0, 1, 90, 172, 30], [11, 26, 7, 3, 1, 289, 36, 13, 33, 0, 1, 2, 1, 0, 1, 90, 172, 30]])
    user_input_array = np.array([])

    # Multi-Selection of Algorithms
    if "Random Forest" in selection_algorithm:
        st.header("Test Result of Random Forest for Employee Absenteeism At Work Dataset")
        RandomForest(user_input_array)
    if "KNN" in selection_algorithm:
        st.header("Test Result of KNN for Employee Absenteeism At Work Dataset")
        KNearestNeighbors(user_input_array)
    if "SVM" in selection_algorithm:
        st.header("Test Result of SVM for Employee Absenteeism At Work Dataset")
        SupportVectorMachine(user_input_array)
    if not selection_algorithm:
        print("Please Select At Least One Algorithm")

# Satisfaction
elif selection_dataset == 'Satisfaction':
    
    # Title of Dataset
    st.header("Employee Satisfaction Dataset")
    st.write("Employee satisfaction reflects the contentment and fulfillment of individuals within an organization."
             "It is a key indicator of workplace morale, productivity, and retention."
             "Understanding and addressing factors that contribute to employee satisfaction are essential for fostering a positive work environment"
             "and achieving organizational success.")

    # Data Reading
    data = pd.read_csv("../Datasets/employee_satisfaction.csv")

    # Display Full Datasets
    st.subheader("Full Dataset of Employee Satisfaction")
    st.write("This dataset contains full dataset of employee satisfaction.")
    data

    # Data Splitting
    data_input = data.drop(columns=['Name', 'Feedback Score', 'Joining Date', 'Satisfaction Rate (%)'])
    data_target = data['Satisfaction Rate (%)']
    x_train, y_train, x_test, y_test = fold_cross_validation (data_input, data_target)

    # Display Both Input and Target Data
    st.subheader("Input Data of Employee Satisfaction")
    st.write("It contains input data of employee absenteeism at work.")
    data_input

    st.subheader("Target Data of Employee Satisfaction")
    st.write("In addition, the target data of employee absenteeism at work is included as well.")
    data_target
    
    # Display Corresponding Training Data
    st.subheader("Input Training Data of Employee Satisfaction")
    st.write("Input training data of employee absenteeism at work is provided below.")
    x_train
    st.subheader("Target Training Data of Employee Satisfaction")
    st.write("Furthermore, target training data of employee absenteeism at work is provided as well.")
    y_train

    # Display Corresponding Testing Data
    st.subheader("Input Testing Data of Employee Satisfaction")
    st.write("Input testing data of employee absenteeism at work is provided below.")
    x_test
    st.subheader("Target Testing Data of Employee Satisfaction")
    st.write("Furthermore, target testing data of employee absenteeism at work is provided as well.")
    y_test

    # Dataset Type
    reg_or_class = 'Regression'
    user_input_array = np.array([])

    # Multi-Selection of Algorithms
    if "Random Forest" in selection_algorithm:
        st.header("Test Result of Random Forest for Employee Satisfaction Dataset")
        RandomForest(user_input_array)
    if "KNN" in selection_algorithm:
        st.header("Test Result of KNN for Employee Satisfaction Dataset")
        KNearestNeighbors(user_input_array)
    if "SVM" in selection_algorithm:
        st.header("Test Result of SVM for Employee Satisfaction Dataset")
        SupportVectorMachine(user_input_array)
    if not selection_algorithm:
        print("Please Select At Least One Algorithm")

# Turnover
elif selection_dataset == 'Turnover':

    # Title of Dataset
    st.header("Employee Turnover Dataset")
    st.write("Employee turnover, often termed attrition, denotes the rate at which employees leave a company and are replaced."
             "It is a critical metric influencing organizational stability, culture, and productivity."
             "Addressing turnover requires strategies for retention, employee engagement, and understanding the factors leading to departures.")

    # Data Reading
    data = pd.read_csv("../Datasets/turnover.csv", encoding='latin1')

    # Display Full Datasets
    st.subheader("Full Dataset of Employee Turnover")
    data
    
    # Assuming you have categorical columns that need encoding
    categorical_columns = ['gender', 'industry', 'profession', 'traffic', 'coach', 'head_gender', 'greywage', 'way']
    data = pd.get_dummies(data, columns=categorical_columns)
    
    # Data Splitting
    data_input = data.drop(columns=['novator', 'event'])
    data_target = data['event']
    x_train, y_train, x_test, y_test = fold_cross_validation (data_input, data_target)

    # Display Both Input and Target Data
    st.subheader("Input Data of Employee Turnover")
    st.write("It contains input data of employee turnover.")
    data_input
    st.subheader("Target Data of Employee Turnover")
    st.write("In addition, the target data of employee turnover is included as well.")
    data_target
    
    # Display Corresponding Training Data
    st.subheader("Input Training Data of Employee Turnover")
    st.write("Input training data of employee turnover is provided below.")
    x_train
    st.subheader("Target Training Data of Employee Turnover")
    st.write("Furthermore, target training data of employee turnover is provided as well.")
    y_train

    # Display Corresponding Testing Data
    st.subheader("Input Testing Data of Employee Turnover")
    st.write("Input testing data of employee turnover is provided below.")
    x_test
    st.subheader("Target Testing Data of Employee Turnover")
    st.write("Furthermore, target testing data of employee turnover is provided as well.")
    y_test
    
    # Dataset Type
    reg_or_class = 'Classification'
    user_input_array = np.array([])

    # Multi-Selection of Algorithms
    if "Random Forest" in selection_algorithm:
        st.header("Test Result of Random Forest for Employee Turnover Dataset")
        RandomForest(user_input_array)
    if "KNN" in selection_algorithm:
        st.header("Test Result of KNN for Employee Turnover Dataset")
        KNearestNeighbors(user_input_array)
    if "SVM" in selection_algorithm:
        st.header("Test Result of SVM for Employee Turnover Dataset")
        SupportVectorMachine(user_input_array)
    if not selection_algorithm:
        print("Please Select At Least One Algorithm")

# Alternative    
else:
    print("Please Select One Dataset")