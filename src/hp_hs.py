import numpy as np
from preprocessing import load_and_preprocess
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import MinMaxScaler

def hp_hs_framework(file_path):
    
    X_train, X_test, y_train, y_test, dynamic_data = load_and_preprocess(file_path)

    # Primary Layer
    knn_scores, svm_scores, knn_acc, svm_acc, hybrid_acc = train_knn_svm(X_train, X_test, y_train, y_test)
    best_weights, primary_apfd, primary_accuracy = random_search_optimize_weights(knn_scores, svm_scores, y_test)

    # Compute Fault_Density for clustering
    dynamic_data['Fault_Density'] = (
        dynamic_data['LastResults_Length'] -
        (dynamic_data['LastResults_Success_Percentage'] * dynamic_data['LastResults_Length'] / 100)
    )

    # Secondary Layer
    dynamic_data, kmeans_model, silhouette_avg = cluster_test_cases(dynamic_data)
    best_q_params, secondary_apfd = random_search_qlearning(dynamic_data)

    # Secondary Layer Metrics
    best_k = kmeans_model.n_clusters  # Retrieve the best number of clusters
    secondary_accuracy = np.mean(dynamic_data['Fault_Density'] > dynamic_data['Fault_Density'].mean())  # Fault detection rate

    # Final Results
    secondary_rankings = q_learning_within_clusters(dynamic_data, **best_q_params)
    final_apfd = calculate_apfd(secondary_rankings, dynamic_data['Fault_Density'])
    final_accuracy = np.mean(dynamic_data.iloc[secondary_rankings]['Fault_Density'] > dynamic_data['Fault_Density'].mean())

    results = {
        "Primary Layer": {
            "KNN Accuracy": knn_acc,
            "SVM Accuracy": svm_acc,
            "Hybrid Accuracy": hybrid_acc,
            "APFD": primary_apfd,
        },
        "Secondary Layer": {
            "Accuracy": secondary_accuracy,
            "Best K": best_k,
            "Silhouette Score": silhouette_avg,
            "Parameters": best_q_params
        },
        "Final Framework": {
            "Accuracy": final_accuracy,
            "APFD": final_apfd
        },
        "Final Rankings": [int(r) for r in secondary_rankings[:10]]
    }
    return results

def train_knn_svm(X_train, X_test, y_train, y_test):
    """
    Train KNN and SVM models and return their prediction probabilities.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    # KNN Grid Search
    knn_params = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')
    knn.fit(X_train, y_train)
    knn_best = knn.best_estimator_
    knn_scores = knn_best.predict_proba(X_test)[:, 1]
    knn_acc = accuracy_score(y_test, knn_best.predict(X_test))

    # SVM Grid Search
    svm_params = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': ['scale', 0.1, 0.01],
        'kernel': ['rbf']
    }
    svm = GridSearchCV(SVC(probability=True), svm_params, cv=5, scoring='accuracy')
    svm.fit(X_train, y_train)
    svm_best = svm.best_estimator_
    svm_scores = svm_best.predict_proba(X_test)[:, 1]
    svm_acc = accuracy_score(y_test, svm_best.predict(X_test))
    
    hybrid_acc = (knn_acc + svm_acc)/2

    return knn_scores, svm_scores, knn_acc, svm_acc, hybrid_acc

def random_search_optimize_weights(knn_scores, svm_scores, y_test, num_samples=100):
    """
    Perform random search to optimize weights for combining KNN and SVM scores.
    """
    best_apfd = -np.inf
    best_accuracy = -np.inf
    best_weights = None

    for _ in range(num_samples):
        w_knn = np.random.uniform(0, 1)
        w_svm = 1 - w_knn

        hybrid_scores = w_knn * knn_scores + w_svm * svm_scores
        rankings = np.argsort(-hybrid_scores)
        apfd = calculate_apfd(rankings, y_test)

        hybrid_predictions = (hybrid_scores > 0.5).astype(int)
        accuracy = accuracy_score(y_test, hybrid_predictions)

        if apfd > best_apfd or (apfd == best_apfd and accuracy > best_accuracy):
            best_apfd = apfd
            best_accuracy = accuracy
            best_weights = (w_knn, w_svm)

    return best_weights, best_apfd, best_accuracy

def cluster_test_cases(dynamic_data):
    """
    Cluster test cases using K-Means and return the clustered data and silhouette score.
    """
    features = ['Duration', 'Cycle', 'Fault_Density']
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(dynamic_data[features])

    best_k = 2
    best_silhouette = -1
    best_model = None

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(normalized_features)
        silhouette_avg = silhouette_score(normalized_features, clusters)
        if silhouette_avg > best_silhouette:
            best_k = k
            best_silhouette = silhouette_avg
            best_model = kmeans

    dynamic_data['Cluster'] = best_model.predict(normalized_features)
    return dynamic_data, best_model, best_silhouette

def random_search_qlearning(dynamic_data, num_iterations=10):
    """
    Perform random search to tune Q-Learning hyperparameters.
    """
    best_params = None
    best_performance = -np.inf

    param_grid = {
        'alpha': np.linspace(0.01, 0.2, 10),
        'gamma': np.linspace(0.8, 1.0, 10),
        'epsilon_decay': np.linspace(0.95, 0.999, 10)
    }

    param_combinations = list(ParameterSampler(param_grid, n_iter=num_iterations, random_state=42))

    for params in param_combinations:
        alpha, gamma, epsilon_decay = params['alpha'], params['gamma'], params['epsilon_decay']
        rankings = q_learning_within_clusters(dynamic_data, alpha, gamma, epsilon_decay)
        apfd = calculate_apfd(rankings, dynamic_data['Fault_Density'])

        if apfd > best_performance:
            best_performance = apfd
            best_params = params

    return best_params, best_performance

def q_learning_within_clusters(dynamic_data, alpha, gamma, epsilon_decay, num_iterations=500):
    """
    Apply Q-Learning to prioritize test cases within each cluster.
    """
    dynamic_data['Q_Rank'] = 0  # Initialize the Q_Rank column

    for cluster_id in dynamic_data['Cluster'].unique():
        # Filter data for the current cluster
        cluster_data = dynamic_data[dynamic_data['Cluster'] == cluster_id].copy()
        num_test_cases = len(cluster_data)
        Q = np.zeros((num_test_cases, num_test_cases))  # Initialize Q-table
        epsilon = 1.0  # Exploration-exploitation trade-off

        for _ in range(num_iterations):
            for state in range(num_test_cases):
                # Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, num_test_cases)  # Explore
                else:
                    action = np.argmax(Q[state])  # Exploit

                # Reward based on Fault Density
                reward = cluster_data.iloc[action]['Fault_Density']

                # Update Q-value
                next_state = (state + 1) % num_test_cases
                Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            epsilon *= epsilon_decay  # Decay epsilon

        # Assign Q-Rank based on the sum of Q-values
        cluster_data.loc[:, 'Q_Rank'] = np.argsort(-Q.sum(axis=1))  # Explicitly use .loc
        dynamic_data.loc[cluster_data.index, 'Q_Rank'] = cluster_data['Q_Rank']  # Update the main DataFrame

    # Sort the entire dataset by Q-Rank for final rankings
    return dynamic_data.sort_values(by='Q_Rank').index


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

