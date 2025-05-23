from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#APFD
def calculate_apfd(probabilities, labels):
    num_test_cases = len(labels)
    num_faults = sum(labels)
    if num_faults == 0:
        print("No faults detected in the dataset. APFD cannot be calculated.")
        return None
    fault_ranks = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
    rank_sum = sum(fault_ranks[:num_faults])
    return 1 - (rank_sum / (num_test_cases * num_faults)) + (1 / (2 * num_test_cases))


# Baseline Models
def primary_layer_baseline_evaluation(X_train, X_test, y_train, y_test):
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),  
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        #probabilities or fallback to decision_function
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            probabilities = model.decision_function(X_test)
        else:
            probabilities = model.predict(X_test) 

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        apfd = calculate_apfd(probabilities, y_test)
        
        #rankings
        results[name] = {
            "accuracy": accuracy,
            "apfd": apfd,
            "ranking": sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True) if probabilities is not None else []
        }
    return results




# Optimized Models
def primary_layer_optimized_evaluation(X_train, X_test, y_train, y_test):
    results = {}

    # KNN Optimization
    knn_optimized = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    knn_optimized.fit(X_train, y_train)
    knn_predictions_opt = knn_optimized.predict(X_test)
    knn_probabilities_opt = knn_optimized.predict_proba(X_test)[:, 1]
    knn_accuracy_opt = accuracy_score(y_test, knn_predictions_opt)
    knn_apfd_opt = calculate_apfd(knn_probabilities_opt, y_test)
    results['KNN'] = {
        "accuracy": knn_accuracy_opt,
        "apfd": knn_apfd_opt,
        "ranking": sorted(range(len(knn_probabilities_opt)), key=lambda i: knn_probabilities_opt[i], reverse=True)
    }

    # Decision Tree Optimization
    dt_optimized = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_optimized.fit(X_train, y_train)
    dt_predictions_opt = dt_optimized.predict(X_test)
    dt_probabilities_opt = dt_optimized.predict_proba(X_test)[:, 1]
    dt_accuracy_opt = accuracy_score(y_test, dt_predictions_opt)
    dt_apfd_opt = calculate_apfd(dt_probabilities_opt, y_test)
    results['Decision Tree'] = {
        "accuracy": dt_accuracy_opt,
        "apfd": dt_apfd_opt,
        "ranking": sorted(range(len(dt_probabilities_opt)), key=lambda i: dt_probabilities_opt[i], reverse=True)
    }

    # Random Forest Optimization
    rf_optimized = RandomForestClassifier(n_estimators=150, random_state=42)
    rf_optimized.fit(X_train, y_train)
    rf_predictions_opt = rf_optimized.predict(X_test)
    rf_probabilities_opt = rf_optimized.predict_proba(X_test)[:, 1]
    rf_accuracy_opt = accuracy_score(y_test, rf_predictions_opt)
    rf_apfd_opt = calculate_apfd(rf_probabilities_opt, y_test)
    feature_importances = rf_optimized.feature_importances_
    results['Random Forest'] = {
        "accuracy": rf_accuracy_opt,
        "apfd": rf_apfd_opt,
        "ranking": sorted(range(len(rf_probabilities_opt)), key=lambda i: rf_probabilities_opt[i], reverse=True),
        "feature_importances": feature_importances.tolist(),
    }

    # SVM Optimization
    svm_optimized = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
    svm_optimized.fit(X_train, y_train)
    svm_predictions_opt = svm_optimized.predict(X_test)
    svm_probabilities_opt = svm_optimized.predict_proba(X_test)[:, 1]
    svm_accuracy_opt = accuracy_score(y_test, svm_predictions_opt)
    svm_apfd_opt = calculate_apfd(svm_probabilities_opt, y_test)
    results['SVM'] = {
        "accuracy": svm_accuracy_opt,
        "apfd": svm_apfd_opt,
        "ranking": sorted(range(len(svm_probabilities_opt)), key=lambda i: svm_probabilities_opt[i], reverse=True)
    }

    return results
