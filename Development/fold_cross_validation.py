import pandas as pd

def fold_cross_validation (data_input, data_target) :
    num_folds = 5
    test_range = int(len(data_input) / num_folds)

    data_input_train = []
    data_target_train = []
    data_input_test = []
    data_target_test = []
    
    for i in range(num_folds):
        # Test set
        test_idx = range(i * test_range, (i + 1) * test_range)
        data_input_test.append(data_input.loc[test_idx])
        data_target_test.append(data_target.loc[test_idx])

        # Train set
        train_idx = [n for n in range(len(data_input)) if n not in test_idx]
        data_input_train.append(data_input.loc[train_idx])
        data_target_train.append(data_target.loc[train_idx])
        
        # Display fold information (optional)
        #print(f"Fold {i + 1}: Train {len(train_idx)}, Test {len(test_idx)}\n")
        
    all_train_input = pd.concat([data_input_train[0], data_input_train[1], data_input_train[2], data_input_train[3], data_input_train[4]])
    all_train_target = pd.concat([data_target_train[0], data_target_train[1], data_target_train[2], data_target_train[3], data_target_train[4]])
    all_test_input = pd.concat([data_input_test[0], data_input_test[1], data_input_test[2], data_input_test[3], data_input_test[4]])
    all_test_target = pd.concat([data_target_test[0], data_target_test[1], data_target_test[2], data_target_test[3], data_target_test[4]])

    return all_train_input, all_train_target, all_test_input, all_test_target