from collections import defaultdict
from preprocessing import load_and_preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterSampler
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler

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

# GridSearch for Tuning Hyperparameters
def tune_knn(X_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best KNN Parameters:", grid.best_params_)
    return grid.best_estimator_, grid.best_params_

def tune_svm(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': ['scale', 0.1],
        'kernel': ['rbf']
    }
    grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best SVM Parameters:", grid.best_params_)
    return grid.best_estimator_, grid.best_params_

# Multi-Objective Optimization Primary Layer
def multi_objective_optimization_primary_layer(X_train, X_test, y_train, y_test):
    # Tune KNN
    knn, knn_best_params = tune_knn(X_train, y_train)
    knn_scores = knn.predict_proba(X_test)[:, 1]
    knn_rankings = np.argsort(-knn_scores)
    knn_apfd = calculate_apfd(knn_rankings, y_test)

    # Tune SVM
    svm, svm_best_params = tune_svm(X_train, y_train)
    svm_scores = svm.predict_proba(X_test)[:, 1]
    svm_accuracy = accuracy_score(y_test, svm.predict(X_test))

    # Final Rankings
    weights = {
        'apfd_weight': 0.7, 
        'accuracy_weight': 0.3  
    }
    combined_scores = (
        weights['apfd_weight'] * knn_scores + 
        weights['accuracy_weight'] * svm_scores
    )
    combined_rankings = np.argsort(-combined_scores)

    # Final Metrics
    knn_accuracy = accuracy_score(y_test, knn.predict(X_test))
    hybrid_predictions = (combined_scores > 0.5).astype(int)  
    hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)  
    primary_apfd = calculate_apfd(combined_rankings, y_test)  
    return knn_rankings, knn_apfd, combined_rankings, hybrid_accuracy, primary_apfd, knn_accuracy, svm_accuracy, knn_best_params, svm_best_params

# Evaluate Rankings
def evaluate_rankings(dynamic_data, rankings):
    labels = dynamic_data['Fault_Density'] > dynamic_data['Fault_Density'].mean()
    labels = labels.astype(int)  

    # APFD
    apfd_score = calculate_apfd(rankings, labels)
    return apfd_score

# Q-Learning Hyperparameter Tuning
def tune_q_learning_hyperparameters(dynamic_data, primary_layer_scores, n_iter=50):
    best_params = None
    best_score = -np.inf
    for _ in range(n_iter):
        alpha = np.random.uniform(0.01, 0.2)
        gamma = np.random.uniform(0.5, 1.0)
        epsilon_decay = np.random.uniform(0.95, 0.999)

        # Evaluate Q-Learning 
        secondary_rankings = q_learning_secondary_layer(dynamic_data, alpha, gamma, epsilon_decay, primary_layer_scores)
        score = evaluate_rankings(dynamic_data, secondary_rankings)

        if score > best_score:
            best_score = score
            best_params = {
                'alpha': alpha,
                'gamma': gamma,
                'epsilon_decay': epsilon_decay
            }

    return best_params

# Q-Learning Secondary Layer
def q_learning_secondary_layer(dynamic_data, alpha, gamma, epsilon_decay, primary_layer_scores, num_iterations=500):
    # Feature Engineering
    dynamic_data['Fault_Density'] = dynamic_data['LastResults_Length'] - (
        dynamic_data['LastResults_Success_Percentage'] * dynamic_data['LastResults_Length']
    )

    # Normalize 
    scaler = MinMaxScaler()
    dynamic_data[['Duration', 'Cycle', 'Fault_Density']] = scaler.fit_transform(
        dynamic_data[['Duration', 'Cycle', 'Fault_Density']]
    )

    # alignment 
    if len(primary_layer_scores) != len(dynamic_data):
        print(f"Aligning in Q-Learning: Primary Scores ({len(primary_layer_scores)}) and Dynamic Data ({len(dynamic_data)})")
        primary_layer_scores = primary_layer_scores[:len(dynamic_data)]

    # Q-table as a dictionary
    Q = defaultdict(lambda: defaultdict(float))

    # Epsilon-greedy params
    epsilon = 1.0

    for _ in range(num_iterations):
        for state_idx, state in enumerate(dynamic_data[['Duration', 'Cycle']].values):
            state_key = tuple(state)
            valid_actions = list(range(len(dynamic_data)))

            
            if random.random() < epsilon:
                action_idx = random.choice(valid_actions)
            else:
                action_idx = max(Q[state_key], key=Q[state_key].get, default=random.choice(valid_actions))

            if action_idx >= len(dynamic_data):
                action_idx = len(dynamic_data) - 1

            # Reward 
            reward = 0.8 * dynamic_data.iloc[action_idx]['Fault_Density'] + 0.2 * primary_layer_scores[action_idx]

            # Next state 
            next_state_idx = (state_idx + 1) % len(dynamic_data)
            next_state_key = tuple(dynamic_data.iloc[next_state_idx][['Duration', 'Cycle']].values)
            max_next_q = max(Q[next_state_key].values(), default=0)
            Q[state_key][action_idx] += alpha * (reward + gamma * max_next_q - Q[state_key][action_idx])

        epsilon *= epsilon_decay

    final_rankings = sorted(
        range(len(dynamic_data)),
        key=lambda idx: -sum(Q.get(tuple(dynamic_data.iloc[idx][['Duration', 'Cycle']].values), {}).values())
    )
    return final_rankings



# Framework Execution
def moo_q_framework(file_path):
    # Preprocessing 
    X_train, X_test, y_train, y_test, dynamic_data = load_and_preprocess(file_path)

    # Primary Layer
    primary_rankings, knn_apfd, combined_rankings, hybrid_acc, primary_apfd, knn_acc, svm_acc, knn_best_params, svm_best_params = multi_objective_optimization_primary_layer(X_train, X_test, y_train, y_test)

    #alignment between combined_rankings and dynamic_data
    if len(combined_rankings) != len(dynamic_data):
        print(f"Aligning sizes: Combined Rankings ({len(combined_rankings)}) and Dynamic Data ({len(dynamic_data)})")
        if len(combined_rankings) > len(dynamic_data):
            combined_rankings = combined_rankings[:len(dynamic_data)]
        else:
            dynamic_data = dynamic_data.iloc[:len(combined_rankings)].reset_index(drop=True)

    # Secondary Layer
    best_params = tune_q_learning_hyperparameters(dynamic_data, primary_layer_scores=combined_rankings, n_iter=10)
    secondary_rankings = q_learning_secondary_layer(dynamic_data, **best_params, primary_layer_scores=combined_rankings)

    # Secondary Layer Accuracy
    predicted_faults = dynamic_data.iloc[secondary_rankings]['Fault_Density'] > dynamic_data['Fault_Density'].mean()
    actual_faults = dynamic_data['Fault_Density'] > dynamic_data['Fault_Density'].mean()
    secondary_accuracy = accuracy_score(actual_faults, predicted_faults)

    final_accuracy = np.mean(predicted_faults)
    final_apfd = calculate_apfd(secondary_rankings, dynamic_data['Fault_Density'])

    results = {
        "Primary Layer": {
            "KNN Accuracy": knn_acc,
            "SVM Accuracy": svm_acc,
            "Hybrid Accuracy": hybrid_acc,
            "APFD": primary_apfd,
            "KNN Best Params": knn_best_params,
            "SVM Best Params": svm_best_params
        },
        "Secondary Layer": {
            "Accuracy": secondary_accuracy,
            "Parameters": best_params
        },
        "Final Framework": {
            "Accuracy": final_accuracy,
            "APFD": final_apfd
        },
        "Final Rankings": [int(r) for r in secondary_rankings[:10]]
    }
    return results

