# Selection Statement On Dataset
from operator import ge
import random
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

# Dropdown List for User Prompt
selection_user_prompt = st.sidebar.selectbox("Input Your Own Data", options = ["Input Own Data", "Yes", "No"])

# --------------------------------------------------- METHOD OF ALGORITHM SELECTION ---------------------------------------------------
def RandomForest(user_input_array):
    if user_input_array.size > 0:
        test_result_type, test_result, user_pred = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 1, np.array(user_input_array).reshape(1, -1))
        st.subheader("Based On User Input [N_Estimation = 1]")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        st.write(f"User Prediction: {user_pred}")
        print("test result (n=1): ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        estimation_options = [1, 5, 10, 15]

        for estimation in estimation_options:
            test_result_type, test_result = random_forest(x_train, x_test, y_train, y_test, reg_or_class, estimation, user_input_array)

            st.subheader(f"Kernel {estimation}")
            
            # Create a list of lists to represent the table data
            table_data = [
                ["Test Result", test_result],
                ["Test Result Type", test_result_type]
            ]

            # Display the table
            st.table(table_data)
        '''
        test_result_type, test_result = random_forest (x_train, x_test, y_train, y_test, reg_or_class, 1, user_input_array.reshape(1, -1))
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
        '''

def KNearestNeighbors(user_input_array):
    if user_input_array.size > 0:
        test_result_type, test_result, user_pred = k_near_neighbor (x_train, x_test, y_train, y_test, reg_or_class, 100, np.array(user_input_array).reshape(1, -1))
        st.subheader("Based On User Input [N_Neighbor = 100]")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        st.write(f"User Prediction: {user_pred}")
        print("test result (n=1): ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        neighbor_options = [100, 200, 300, 400]

        for neighbor in neighbor_options:
            test_result_type, test_result = support_vector_machine(x_train, x_test, y_train, y_test, reg_or_class, neighbor, user_input_array)

            st.subheader(f"Kernel {neighbor}")
            
            # Create a list of lists to represent the table data
            table_data = [
                ["Test Result", test_result],
                ["Test Result Type", test_result_type]
            ]

            # Display the table
            st.table(table_data)
        
        '''
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
        '''

def SupportVectorMachine(user_input_array):
    if user_input_array.size > 0:
        test_result_type, test_result, user_pred = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'rbf', np.array(user_input_array).reshape(1, -1))
        st.subheader("Based On User Input [Kernel = RBF]")
        st.write(f"Test Result: {test_result} [{test_result_type}]")
        st.write(f"User Prediction: {user_pred}")
        print("test result (n=1): ", test_result, "(", test_result_type, ")")
        print("user_pred : ", user_pred)
    else:
        kernel_options = ['linear', 'rbf', 'sigmoid', 'poly']

        for kernel in kernel_options:
            test_result_type, test_result = support_vector_machine(x_train, x_test, y_train, y_test, reg_or_class, kernel, user_input_array)

            st.subheader(f"Kernel {kernel.capitalize()}")
            
            # Create a list of lists to represent the table data
            table_data = [
                ["Test Result", test_result],
                ["Test Result Type", test_result_type]
            ]

            # Display the table
            st.table(table_data)
        
        #test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'linear', user_input_array)
        #st.subheader("Kernel Linear")
        #st.write(f"Test Result: {test_result} [{test_result_type}]")
        #print("test result (linear): ", test_result, "(", test_result_type, ")")
        #test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'rbf', user_input_array)
        #st.subheader("Kernel RBF")
        #st.write(f"Test Result: {test_result} [{test_result_type}]")
        #print("test result (rbf): ", test_result, "(", test_result_type, ")")
        #test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'sigmoid', user_input_array)
        #st.subheader("Kernel Sigmoid")
        #st.write(f"Test Result: {test_result} [{test_result_type}]")
        #print("test result (sigmoid): ", test_result, "(", test_result_type, ")")
        #test_result_type, test_result = support_vector_machine (x_train, x_test, y_train, y_test, reg_or_class, 'poly', user_input_array)
        #st.subheader("Kernel Poly")
        #st.write(f"Test Result: {test_result} [{test_result_type}]")
        #print("test result (  poly ): ", test_result, "(", test_result_type, ")")


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
    st.write("As provided below, this dataset is specifically used for the input of training data. Additionally, all unnecessary columns are dropped.")
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

    if selection_user_prompt == "Yes":
        designation = st.sidebar.slider("Select Designation", min_value=0, max_value=10, value=0)
        resource = st.sidebar.slider("Select Resource Allocation", min_value=0, max_value=10, value=0)
        gender = st.sidebar.selectbox("Select Gender", options = ["Select Gender", 'Male', 'Female'])
        companyType = st.sidebar.selectbox("Select Company Type", options = ["Select Type", 'Product', 'Service'])
        wfh = st.sidebar.selectbox("Select WFH Setup Available", options = ["Select Availability", 'Yes', 'No'])

        # Map categorical variables to numeric representations
        gender_female = 1 if gender == 'Female' else 0
        gender_male = 1 if gender == 'Male' else 0
        companyType_service = 1 if companyType == 'Service' else 0
        companyType_product = 1 if companyType == 'Product' else 0
        wfh_yes = 1 if wfh == 'Yes' else 0
        wfh_no = 1 if wfh == 'No' else 0

        # User Prompt
        user_input_array = np.array([
            designation, resource, gender_female, gender_male, companyType_product, companyType_service, wfh_no, wfh_yes
        ], dtype=object)

        # Multi-Selection of Algorithms
        for algorithm in selection_algorithm:
            if algorithm == "Random Forest":
                st.header("Test Result of Random Forest for Employee Burnout Dataset")
                RandomForest(user_input_array)
            elif algorithm == "KNN":
                st.header("Test Result of KNN for Employee Burnout Dataset")
                KNearestNeighbors(user_input_array)
            elif algorithm == "SVM":
                st.header("Test Result of SVM for Employee Burnout Dataset")
                SupportVectorMachine(user_input_array)
            else:
                st.warning("Please Select One Algorithm")
    elif selection_user_prompt == "No":
        user_input_array = np.array([])

        for algorithm in selection_algorithm:
            if algorithm == "Random Forest":
                st.header("Test Result of Random Forest for Employee Burnout Dataset")
                RandomForest(user_input_array)
            elif algorithm == "KNN":
                st.header("Test Result of KNN for Employee Burnout Dataset")
                KNearestNeighbors(user_input_array)
            elif algorithm == "SVM":
                st.header("Test Result of SVM for Employee Burnout Dataset")
                SupportVectorMachine(user_input_array)
            else:
                st.warning("Please Select One Algorithm")
    else:
        print("None")

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
    data_input = data.drop(columns=['Reason for absence', 'Seasons', 'Work load Average/day ', 'Hit target', 'Absenteeism time in hours'])
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
    
    if selection_user_prompt == "Yes":
        id = st.sidebar.text_input("Insert ID")
        month_absence = st.sidebar.slider("Select Month of Absence", min_value=0, max_value=12, value=0)
        day_of_week = st.sidebar.selectbox("Select Day of Week", options = ["Select Day", 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
        transportation_expense = st.sidebar.slider("Select Transportation Expenses", min_value=0, max_value=500, value=0)  
        distance = st.sidebar.slider("Select Distance From Residence to Work", min_value=0, max_value=100, value=0)  
        service_time = st.sidebar.slider("Select Service Time", min_value=0, max_value=20, value=0) 
        age = st.sidebar.slider("Select Age", min_value=0, max_value=60, value=0)  
        disciplinary_failure = st.sidebar.selectbox("Select Disciplinary Failure", options = ["Select Type of Disciplinary Failure", 'Yes', 'No']) 
        education = st.sidebar.selectbox("Select Education", options = ["Select Education", 'High School', 'Graduate', 'Postgraduate', 'Doctor']) 
        son = st.sidebar.slider("Select Number of Children", min_value=0, max_value=10, value=0)  
        drinker = st.sidebar.selectbox("Select Type of Social Drinker", options = ["Select Type of Social Drinker", 'Yes', 'No']) 
        smoker = st.sidebar.selectbox("Select Type of Social Smoker", options = ["Select Type of Social Smoker", 'Yes', 'No']) 
        pet = st.sidebar.slider("Select Number of Pet", min_value=0, max_value=10, value=0)  
        weight = st.sidebar.text_input("Insert weight")
        height = st.sidebar.text_input("Insert height (cm)")
        
        try:
            id = float(id)
            weight = float(weight)
            height = float(height)
            height = height/100
            # Check for non-zero height to avoid division by zero
            if height != 0:
                bmi = weight / (height * height)
            else:
                st.sidebar.warning("Height should be non-zero for BMI calculation.")
                bmi = None
        except ValueError:
            st.sidebar.warning("Please enter valid numerical values for weight and height.")
            bmi = None

        # Display BMI
        display_bmi = st.sidebar.text_input("BMI", bmi)

        # Map categorical variables to numeric representations
        day_week = 2 if day_of_week == 'Monday' else 0
        day_week = 3 if day_of_week == 'Tuesday' else 0
        day_week = 4 if day_of_week == 'Wednesday' else 0
        day_week = 5 if day_of_week == 'Thursday' else 0
        day_week = 6 if day_of_week == 'Friday' else 0

        discipline = 1 if disciplinary_failure == 'Yes' else 0
        discipline = 0 if disciplinary_failure == 'No' else 1

        education_level = 1 if education == 'High School' else 0
        education_level = 2 if education == 'Graduate' else 0
        education_level = 3 if education == 'Postgraduate' else 0
        education_level = 4 if education == 'Doctor' else 0

        social_drinker = 1 if drinker == 'Yes' else 0
        social_drinker = 0 if drinker == 'No' else 1

        social_smoker = 1 if smoker == 'Yes' else 0
        social_smoker = 0 if smoker == 'No' else 1

        # User Prompt
        user_input_array = np.array([
            id, month_absence, day_of_week, transportation_expense, distance, 
            service_time, age, disciplinary_failure, education, son, drinker, 
            smoker, pet, weight, height, display_bmi
        ], dtype=object)

        # Ensure that numerical values are used for the model
        user_input_array = np.array([float(val) if isinstance(val, (int, float)) else 0.0 for val in user_input_array])

        # Reshape to (1, -1)
        user_input_array = np.reshape(user_input_array, (1, -1))

        # Multi-Selection of Algorithms
        for algorithm in selection_algorithm:
            if algorithm == "Random Forest":
                st.header("Test Result of Random Forest for Employee Absenteeism Dataset")
                RandomForest(user_input_array)
            elif algorithm == "KNN":
                st.header("Test Result of KNN for Employee Absenteeism Dataset")
                KNearestNeighbors(user_input_array)
            elif algorithm == "SVM":
                st.header("Test Result of SVM for Employee Absenteeism Dataset")
                SupportVectorMachine(user_input_array)
            else:
                st.warning("Please Select One Algorithm")
    elif selection_user_prompt == "No":
        user_input_array = np.array([])

        for algorithm in selection_algorithm:
            if algorithm == "Random Forest":
                st.header("Test Result of Random Forest for Employee Absenteeism Dataset")
                RandomForest(user_input_array)
            elif algorithm == "KNN":
                st.header("Test Result of KNN for Employee Absenteeism Dataset")
                KNearestNeighbors(user_input_array)
            elif algorithm == "SVM":
                st.header("Test Result of SVM for Employee Absenteeism Dataset")
                SupportVectorMachine(user_input_array)
            else:
                st.warning("Please Select One Algorithm")
    else:
        print("None")

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

    # Dummies
    categorical_columns = ['Gender', 'Department', 'Position']
    data = pd.get_dummies(data, columns=categorical_columns)

    # Data Splitting
    data_input = data.drop(columns=['Name', 'Feedback Score', 'Joining Date', 'Satisfaction Rate (%)'])
    data_target = data['Satisfaction Rate (%)']
    x_train, y_train, x_test, y_test = fold_cross_validation (data_input, data_target)

    # Display Both Input and Target Data
    st.subheader("Input Data of Employee Satisfaction")
    st.write("It contains input data of employee satisfaction at work.")
    data_input

    st.subheader("Target Data of Employee Satisfaction")
    st.write("In addition, the target data of employee satisfaction is included as well.")
    data_target
    
    # Display Corresponding Training Data
    st.subheader("Input Training Data of Employee Satisfaction")
    st.write("Input training data of employee satisfaction is provided below.")
    x_train
    st.subheader("Target Training Data of Employee Satisfaction")
    st.write("Furthermore, target training data of employee satisfaction is provided as well.")
    y_train

    # Display Corresponding Testing Data
    st.subheader("Input Testing Data of Employee Satisfaction")
    st.write("Input testing data of employee satisfaction is provided below.")
    x_test
    st.subheader("Target Testing Data of Employee Satisfaction")
    st.write("Furthermore, target testing data of employee satisfaction is provided as well.")
    y_test

    # Dataset Type
    reg_or_class = 'Regression'

    if selection_user_prompt == "Yes":
        age = st.sidebar.slider("Select Age", min_value=0, max_value=60, value=0)  
        projects_completed = st.sidebar.slider("Select Projects Completed", min_value=0, max_value=30, value=0)  
        productivity = st.sidebar.slider("Select Productivity", min_value=0, max_value=100, value=0) 
        salary = st.sidebar.text_input("Insert Salary")
        gender = st.sidebar.selectbox("Select Gender", options = ["Select Gender", 'Male', 'Female'])
        department = st.sidebar.selectbox("Select Department", options = ["Select Department", 'Finance', 'HR', 'IT', 'Marketing', 'Sales'])
        position = st.sidebar.selectbox("Select Position", options = ["Select Position", 'Analyst', 'Intern', 'Junior Developer', 'Manager', 'Senior Developer', 'Team Lead'])

        # Map categorical variables to numeric representations
        gender_female = 1 if gender == 'Female' else 0
        gender_male = 1 if gender == 'Male' else 0

        department_finance = 1 if department == "Finance" else 0
        department_hr = 1 if department == "HR" else 0
        department_it = 1 if department == "IT" else 0
        department_marketing = 1 if department == "Marketing" else 0
        department_sales = 1 if department == "Sales" else 0

        position_analyst = 1 if position == "Analyst" else 0
        position_intern = 1 if position == "Intern" else 0
        position_junior = 1 if position == "Junior Developer" else 0
        position_manager = 1 if position == "Manager" else 0
        position_senior = 1 if position == "Senior Developer" else 0
        position_lead = 1 if position == "Team Lead" else 0

        # User Prompt
        user_input_array = np.array([
            age, projects_completed, productivity, salary, gender_female, gender_male, 
            department_finance, department_hr, department_it, department_marketing, department_sales,
            position_analyst, position_intern, position_junior, position_manager, position_senior, position_lead
        ], dtype=object)

        # Ensure that numerical values are used for the model
        user_input_array = np.array([float(val) if isinstance(val, (int, float)) else 0.0 for val in user_input_array])

        # Reshape to (1, -1)
        user_input_array = np.reshape(user_input_array, (1, -1))

        # Multi-Selection of Algorithms
        for algorithm in selection_algorithm:
            if algorithm == "Random Forest":
                st.header("Test Result of Random Forest for Employee Satisfaction Dataset")
                RandomForest(user_input_array)
            elif algorithm == "KNN":
                st.header("Test Result of KNN for Employee Satisfaction Dataset")
                KNearestNeighbors(user_input_array)
            elif algorithm == "SVM":
                st.header("Test Result of SVM for Employee Satisfaction Dataset")
                SupportVectorMachine(user_input_array)
            else:
                st.warning("Please Select One Algorithm")
    elif selection_user_prompt == "No":
        user_input_array = np.array([])

        for algorithm in selection_algorithm:
            if algorithm == "Random Forest":
                st.header("Test Result of Random Forest for Employee Satisfaction Dataset")
                RandomForest(user_input_array)
            elif algorithm == "KNN":
                st.header("Test Result of KNN for Employee Satisfaction Dataset")
                KNearestNeighbors(user_input_array)
            elif algorithm == "SVM":
                st.header("Test Result of SVM for Employee Satisfaction Dataset")
                SupportVectorMachine(user_input_array)
            else:
                st.warning("Please Select One Algorithm")
    else:
        print("None")

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
    
    if selection_user_prompt == "Yes":
        stag = st.sidebar.slider("Rate Experience", min_value=0.0, max_value=50.0, value=0.0, step=0.01)  
        age = st.sidebar.slider("Select Age", min_value=0, max_value=60, value=0)  
        extraversion = st.sidebar.slider("Rate Extraversion", min_value=0.0, max_value=10.0, value=0.0, step=0.1)  
        independ = st.sidebar.slider("Rate Independ", min_value=0.0, max_value=10.0, value=0.0, step=0.1)  
        selfcontrol = st.sidebar.slider("Rate Self Control", min_value=0.0, max_value=10.0, value=0.0, step=0.1)  
        anxiety = st.sidebar.slider("Rate Anxiety", min_value=0.0, max_value=10.0, value=0.0, step=0.1)  
        gender = st.sidebar.selectbox("Select Gender", options = ["Select Gender", 'Male', 'Female'])

        industry = st.sidebar.selectbox("Select Industry", options = ["Select Industry", 'HoReCa', 'Agriculture', 'Bank', 'Building', 'Consult', 'IT', 'Mining', 'Pharma', 
                                                                      'Power Generation', 'Real Estate', 'Retail', 'State', 'Telecom', 'etc', 'Manufacture', 'Transport'])
        profession = st.sidebar.selectbox("Select Profession", options = ["Select Profession", 'Accounting', 'Business Development', 'Commercial', 'Consult', 'Engineer', 'Finance',
                                                                          'HR', 'IT', 'Law', 'Marketing', 'PR', 'Sales', 'Teaching', 'etc', 'Manage'])
        traffic = st.sidebar.selectbox("Select Traffic", options = ["Select Traffic", 'KA', 'Advert', 'empjs', 'Friends', 'rabrecNErab', 'recNErab', 'Referal', 'Youjs'])
        coach = st.sidebar.selectbox("Select Coach", options = ["Select Coach", 'My Head', 'No', 'Yes'])
        head_gender = st.sidebar.selectbox("Select Head Gender", options = ["Select Head Gender", 'Female', 'Male'])
        greywage = st.sidebar.selectbox("Select Greywage", options = ["Select Greywage", 'Grey', 'White'])
        way = st.sidebar.selectbox("Select Way", options = ["Select Way", 'Bus', 'Car', 'Foot'])


        # Map categorical variables to numeric representations
        gender_female = 1 if gender == 'Female' else 0
        gender_male = 1 if gender == 'Male' else 0

        industry_horeca = 1 if industry == "HoReCa" else 0
        industry_agriculture = 1 if industry == "Agriculture" else 0
        industry_bank = 1 if industry == "Bank" else 0
        industry_buidling = 1 if industry == "Building" else 0
        industry_consult = 1 if industry == "Consult" else 0
        industry_it = 1 if industry == "IT" else 0
        industry_mining = 1 if industry == "Mining" else 0
        industry_pharma = 1 if industry == "Pharma" else 0
        industry_power = 1 if industry == "Power Generation" else 0
        industry_estate = 1 if industry == "Real Estate" else 0
        industry_retail = 1 if industry == "Retail" else 0
        industry_state = 1 if industry == "State" else 0
        industry_telecom = 1 if industry == "Telecom" else 0
        industry_etc = 1 if industry == "etc" else 0
        industry_manufacture = 1 if industry == "Manufacture" else 0
        industry_transport = 1 if industry == "Transport" else 0

        profession_accounting = 1 if profession == "Accounting" else 0
        profession_business = 1 if profession == "Business Development" else 0
        profession_commercial = 1 if profession == "Commercial" else 0
        profession_consult= 1 if profession == "Consult" else 0
        profession_engineer = 1 if profession == "Engineer" else 0
        profession_finance = 1 if profession == "Finance" else 0
        profession_hr = 1 if profession == "HR" else 0
        profession_it= 1 if profession == "IT" else 0
        profession_law = 1 if profession == "Law" else 0
        profession_marketing = 1 if profession == "Marketing" else 0
        profession_pr = 1 if profession == "PR" else 0
        profession_sales = 1 if profession == "Sales" else 0
        profession_teaching = 1 if profession == "Teaching" else 0
        profession_etc = 1 if profession == "etc" else 0
        profession_manage = 1 if profession == "Manage" else 0

        traffic_ka = 1 if traffic == "KA" else 0
        traffic_advert = 1 if traffic == "Advert" else 0
        traffic_empjs = 1 if traffic == "empjs" else 0
        traffic_friends = 1 if traffic == "Friends" else 0
        traffic_rab = 1 if traffic == "rabrecNErab" else 0
        traffic_rec = 1 if traffic == "recNErab" else 0
        traffic_ref = 1 if traffic == "Referal" else 0
        traffic_youjs = 1 if traffic == "Youjs" else 0

        coach_head = 1 if coach == "My Head" else 0
        coach_no = 1 if coach == "No" else 0
        coach_yes = 1 if coach == "Yes" else 0

        head_female = 1 if head_gender == "Female" else 0
        head_male = 1 if head_gender == "Male" else 0

        greywage_grey = 1 if greywage == "Grey" else 0
        greywage_white = 1 if greywage == "White" else 0

        way_bus = 1 if way == "Bus" else 0
        way_car = 1 if way == "Car" else 0
        way_foot = 1 if way == "Foot" else 0

        # User Prompt
        user_input_array = np.array([
            stag, age, extraversion, independ, selfcontrol, anxiety, gender_female, gender_male, industry_horeca, industry_agriculture, industry_bank, 
            industry_buidling, industry_consult, industry_it, industry_mining, industry_pharma, industry_power, industry_estate, industry_retail, industry_state, industry_telecom, industry_etc,
            industry_manufacture, industry_transport, profession_accounting, profession_business, profession_commercial, profession_consult, profession_engineer,
            profession_finance, profession_hr, profession_it, profession_law, profession_marketing, profession_pr, profession_sales, profession_teaching, profession_etc, profession_manage,
            traffic_ka, traffic_advert, traffic_empjs, traffic_friends, traffic_rab, traffic_rec, traffic_ref, traffic_youjs, coach_head, coach_no, coach_yes, head_female,
            head_male, greywage_grey, greywage_white, way_bus, way_car, way_foot
        ], dtype=object)

        
        # Ensure that numerical values are used for the model
        user_input_array = np.array([float(val) if isinstance(val, (int, float)) else 0.0 for val in user_input_array])

        # Reshape to (1, -1)
        user_input_array = np.reshape(user_input_array, (1, -1))

        # Multi-Selection of Algorithms
        for algorithm in selection_algorithm:
            if algorithm == "Random Forest":
                st.header("Test Result of Random Forest for Employee Turnover Dataset")
                RandomForest(user_input_array)
            elif algorithm == "KNN":
                st.header("Test Result of KNN for Employee Turnover Dataset")
                KNearestNeighbors(user_input_array)
            elif algorithm == "SVM":
                st.header("Test Result of SVM for Employee Turnover Dataset")
                SupportVectorMachine(user_input_array)
            else:
                st.warning("Please Select One Algorithm")
    elif selection_user_prompt == "No":
        user_input_array = np.array([])

        for algorithm in selection_algorithm:
            if algorithm == "Random Forest":
                st.header("Test Result of Random Forest for Employee Turnover Dataset")
                RandomForest(user_input_array)
            elif algorithm == "KNN":
                st.header("Test Result of KNN for Employee Turnover Dataset")
                KNearestNeighbors(user_input_array)
            elif algorithm == "SVM":
                st.header("Test Result of SVM for Employee Turnover Dataset")
                SupportVectorMachine(user_input_array)
            else:
                st.warning("Please Select One Algorithm")
    else:
        print("None")

# Alternative    
else:
    print("Please Select One Dataset")