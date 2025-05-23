from preprocessing import load_and_preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np

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

def calculate_apfd_with_density(rankings, fault_density_values):
    # Normalize fault density to 0 and 1
    fault_density_values = fault_density_values / fault_density_values.max()

    # alignment between rankings and fault density
    fault_density_values = fault_density_values[rankings]

    # Calculate fault positions based on rankings
    fault_positions = [(i + 1) * fault_density_values[i] for i in range(len(fault_density_values))]
    num_faults = fault_density_values.sum() 
    num_test_cases = len(fault_density_values)

    # APFD 
    apfd = 1 - (sum(fault_positions) / (num_test_cases * num_faults)) + (1 / (2 * num_test_cases))
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

# Secondary Layer: Optimized K-Means
def optimized_kmeans(dynamic_data):
    # Feature Engineering
    dynamic_data['Fault_Density'] = dynamic_data['LastResults_Length'] - (
        dynamic_data['LastResults_Success_Percentage'] * dynamic_data['LastResults_Length'])
    dynamic_data['Time_Deviation'] = dynamic_data['Duration'] - dynamic_data['Duration'].mean()

    # Normalize 
    features = ['Duration', 'Cycle', 'LastResults_Success_Percentage', 'LastResults_Length', 'Fault_Density',
                'Time_Deviation']
    weights = np.array([0.2, 0.2, 0.3, 0.1, 0.1, 0.1]) 
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(dynamic_data[features])
    weighted_data = normalized_data * weights

    # Optimize K
    best_k = 2
    best_silhouette = -1
    best_model = None

    for k in range(2, 10): 
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(weighted_data)
        silhouette_avg = silhouette_score(weighted_data, clusters)
        if silhouette_avg > best_silhouette:
            best_k = k
            best_silhouette = silhouette_avg
            best_model = kmeans

    # Assigning clusters
    clusters = best_model.predict(weighted_data)
    dynamic_data['Cluster'] = clusters

    # Accuracy: Compare cluster fault density with actual Verdicts
    cluster_fault_density = dynamic_data.groupby('Cluster')['Verdict'].mean()
    predicted_faults = dynamic_data['Cluster'].map(lambda x: 1 if cluster_fault_density[x] > 0.5 else 0)
    accuracy = accuracy_score(dynamic_data['Verdict'], predicted_faults)

    # Rank clusters by Duration and prioritize test cases
    cluster_ranks = dynamic_data.groupby('Cluster')['Duration'].mean().sort_values().index
    prioritized_indices = dynamic_data.sort_values(
        by=['Cluster', 'Duration'],
        key=lambda x: x.map({rank: i for i, rank in enumerate(cluster_ranks)})
    ).index

    return prioritized_indices, best_k, best_silhouette, accuracy

# Framework
def svm_knn_km_framework(file_path):
    # Preprocessing
    X_train, X_test, y_train, y_test, dynamic_data = load_and_preprocess(file_path)

    # Primary Layer
    primary_rankings, combined_scores, knn_acc, svm_acc, hybrid_acc, knn_best_params, svm_best_params = hybrid_primary_layer(X_train, X_test, y_train, y_test)
    primary_apfd = calculate_apfd(primary_rankings, y_test)
    
    if 'Hybrid_Score' not in dynamic_data.columns:
        dynamic_data['Hybrid_Score'] = np.nan  #NaN to ensure float64 type

    dynamic_data.loc[X_test.index, 'Hybrid_Score'] = combined_scores

    #casting column to float64
    dynamic_data['Hybrid_Score'] = dynamic_data['Hybrid_Score'].astype(float)

    # Secondary Layer
    secondary_rankings, best_k, silhouette_score, secondary_accuracy = optimized_kmeans(dynamic_data)

    # Final Framework Accuracy
    cluster_fault_density = dynamic_data.groupby('Cluster')['Verdict'].mean()
    predicted_faults = dynamic_data['Cluster'].map(lambda x: 1 if cluster_fault_density[x] > 0.5 else 0)
    final_accuracy_secondary = accuracy_score(dynamic_data['Verdict'], predicted_faults)
    final_accuracy = (hybrid_acc + final_accuracy_secondary) / 2

    # Final Framework APFD
    combined_scores = (
        0.5 * dynamic_data.loc[secondary_rankings, 'Hybrid_Score'] + 
        0.5 * dynamic_data['Cluster'].map(lambda x: cluster_fault_density[x])
    )
    combined_scores = np.nan_to_num(combined_scores, nan=0.0)
    adjusted_rankings = np.argsort(-combined_scores)
    fault_density_values = dynamic_data.loc[adjusted_rankings, 'Fault_Density'].values
    final_apfd = calculate_apfd_with_density(adjusted_rankings, fault_density_values)

    # Results
    results = {
        "Primary Layer": {
            "KNN Accuracy": knn_acc,
            "KNN Best Params": knn_best_params,
            "SVM Accuracy": svm_acc,
            "Hybrid Accuracy": hybrid_acc,
            "APFD": primary_apfd
        },
        "Secondary Layer": {
            "Accuracy": secondary_accuracy,
            "Best K": best_k,
            "Silhouette Score": silhouette_score,
        },
        "Final Framework": {
            "Accuracy": final_accuracy,
            "APFD": final_apfd
        },
        "Final Rankings": list(secondary_rankings)[:10],  
    }
    return results
