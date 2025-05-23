from preprocessing import load_and_preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterSampler
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

# APFD 
def calculate_apfd(rankings, labels):
    rankings = np.array(rankings)
    labels = np.array(labels)
    num_test_cases = len(labels)
    num_faults = np.sum(labels)
    if num_faults == 0:
        return 0  
    fault_positions = [i + 1 for i in rankings if labels[i] == 1]
    apfd = 1 - (np.sum(fault_positions) / (num_test_cases * num_faults)) + (1 / (2 * num_test_cases))
    return apfd

# Tune SVM 
def tune_svm(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': ['scale', 0.1, 0.01],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print("Best SVM Parameters:", grid.best_params_)
    return grid.best_estimator_, grid.best_params_

# Tune KNN 
def tune_knn(X_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print("Best KNN Parameters:", grid.best_params_)
    return grid.best_estimator_, grid.best_params_

# Primary Layer: Hybrid Feature-Based (KNN + SVM)
def hybrid_primary_layer(X_train, X_test, y_train, y_test):
    # Tune KNN
    knn, knn_best_params = tune_knn(X_train, y_train)
    knn_scores = knn.predict_proba(X_test)[:, 1]

    # Tune SVM
    svm, svm_best_params = tune_svm(X_train, y_train)
    svm_scores = svm.predict_proba(X_test)[:, 1]

    combined_scores = 0.5 * knn_scores + 0.5 * svm_scores
    rankings = np.argsort(-combined_scores)

    # Evaluate
    knn_accuracy = accuracy_score(y_test, knn.predict(X_test))
    svm_accuracy = accuracy_score(y_test, svm.predict(X_test))
    hybrid_predictions = (combined_scores > 0.5).astype(int)
    hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)

    return rankings, combined_scores, knn_accuracy, svm_accuracy, hybrid_accuracy, knn_best_params, svm_best_params

# Random Search for Q-Learning
def tune_q_learning_hyperparameters(dynamic_data, n_iter=10):
    param_grid = {
        'alpha': [0.01, 0.05, 0.1, 0.2],
        'gamma': [0.8, 0.9, 0.95, 0.99],
        'epsilon_decay': [0.999, 0.9995, 0.9999]
    }

    best_params = None
    best_performance = -np.inf
    param_combinations = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

    for params in param_combinations:
        alpha, gamma, epsilon_decay = params['alpha'], params['gamma'], params['epsilon_decay']
        performance = evaluate_q_learning(dynamic_data, alpha, gamma, epsilon_decay)
        if performance > best_performance:
            best_performance = performance
            best_params = params

    print("Best Q-Learning Parameters:", best_params)
    return best_params

# Evaluation 
def evaluate_q_learning(dynamic_data, alpha, gamma, epsilon_decay):
    rankings = q_learning_secondary_layer(dynamic_data, alpha, gamma, epsilon_decay)
    top_half_size = len(dynamic_data) // 2
    faults_detected = dynamic_data.iloc[rankings]['Fault_Density'].values[:top_half_size]
    performance = np.mean(faults_detected)
    return performance

# Secondary Layer: Q-Learning
def q_learning_secondary_layer(dynamic_data, alpha, gamma, epsilon_decay, num_iterations=500):
    # Feature Engineering
    dynamic_data['Fault_Density'] = dynamic_data['LastResults_Length'] - (
        dynamic_data['LastResults_Success_Percentage'] * dynamic_data['LastResults_Length']
    )

    # Normalize 
    scaler = MinMaxScaler()
    dynamic_data[['Duration', 'Cycle', 'Fault_Density']] = scaler.fit_transform(
        dynamic_data[['Duration', 'Cycle', 'Fault_Density']]
    )

    # Q-table as a dictionary
    Q = defaultdict(lambda: defaultdict(float))

    # Epsilon-greedy params
    epsilon = 1.0

    for _ in range(num_iterations):
        for state_idx, state in enumerate(dynamic_data[['Duration', 'Cycle']].values):
            if random.random() < epsilon:
                action_idx = random.randint(0, len(dynamic_data) - 1)
            else:
                #Best action from Q-table
                action_idx = max(Q[state_idx], key=Q[state_idx].get, default=random.randint(0, len(dynamic_data) - 1))

            # Rewards
            reward = dynamic_data.iloc[action_idx]['Fault_Density']

            #update state
            next_state_idx = (state_idx + 1) % len(dynamic_data)
            next_state = dynamic_data.iloc[next_state_idx][['Duration', 'Cycle']].values
            max_next_q = max(Q[next_state_idx].values(), default=0)
            Q[state_idx][action_idx] += alpha * (reward + gamma * max_next_q - Q[state_idx][action_idx])

        epsilon *= epsilon_decay

    #rankings on Q-values
    final_rankings = sorted(range(len(dynamic_data)), key=lambda idx: -sum(Q[idx].values()))
    return final_rankings

# Framework Execution
def knn_svm_q_framework(file_path):
    # Preprocessing
    X_train, X_test, y_train, y_test, dynamic_data = load_and_preprocess(file_path)

    # Primary Layer
    primary_rankings, combined_scores, knn_acc, svm_acc, hybrid_acc, knn_best_params, svm_best_params = hybrid_primary_layer(X_train, X_test, y_train, y_test)
    primary_apfd = calculate_apfd(primary_rankings, y_test)

    # Secondary Layer
    best_params = tune_q_learning_hyperparameters(dynamic_data, n_iter=10)
    secondary_rankings = q_learning_secondary_layer(dynamic_data, **best_params)

    # Secondary Layer Accuracy
    predicted_faults = dynamic_data.iloc[secondary_rankings]['Fault_Density'] > dynamic_data['Fault_Density'].mean()
    actual_faults = dynamic_data['Fault_Density'] > dynamic_data['Fault_Density'].mean()
    secondary_accuracy = accuracy_score(actual_faults, predicted_faults)


    # Final Framework Accuracy and APFD
    predicted_faults = dynamic_data.iloc[secondary_rankings]['Fault_Density'] > dynamic_data['Fault_Density'].mean()
    final_accuracy = np.mean(predicted_faults)
    final_apfd = calculate_apfd(secondary_rankings, dynamic_data['Fault_Density'])

    # Results
    results = {
        "Primary Layer": {
            "KNN Accuracy": knn_acc,
            "SVM Accuracy": svm_acc,
            "Hybrid Accuracy": hybrid_acc,
            "APFD": primary_apfd
        },
        "Secondary Layer": {
            "Accuracy": secondary_accuracy,
            "Parameters": best_params,
        },
        "Final Framework": {
            "Accuracy": final_accuracy,
            "APFD": final_apfd
        },
        "Final Rankings": [int(r) for r in secondary_rankings[:10]] 
    }
    return results
