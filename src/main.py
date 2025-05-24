from preprocessing import load_and_preprocess
from SVM_KNN_KM import svm_knn_km_framework
from KNN_SVM_Q import knn_svm_q_framework
from moo_km import moo_km_framework
from moo_q import moo_q_framework
from hp_hs import hp_hs_framework
from primary_layer import (
    primary_layer_baseline_evaluation,
    primary_layer_optimized_evaluation
)
from secondary_layer import (
    kmeans_baseline_evaluation,
    kmeans_optimized_evaluation,
    q_learning_baseline_evaluation,
    q_learning_optimized_evaluation
)

def print_results(title, results, include_feature_importances=False):
    print(f"\n--- {title} ---")
    for model, metrics in results.items():
        if include_feature_importances and "feature_importances" in metrics:
            print(f"{model} - Accuracy: {metrics['accuracy']:.4f}, APFD: {metrics['apfd']:.4f}, "
                  f"Feature Importances: {metrics['feature_importances']}")
        else:
            print(f"{model} - Accuracy: {metrics['accuracy']:.4f}, APFD: {metrics['apfd']:.4f}")

def print_secondary_results(title, results):
    print(f"\n--- {title} ---")
    print(f"APFD: {results['APFD']:.4f}")
    print(f"Execution Time: {results['Execution_Time']:.4f}s")
    print(f"Cost per Fault Detected: {results['Cost_Per_Fault']:.4f}")
    if "Number_of_Clusters" in results:  
        print(f"Number of Clusters: {results['Number_of_Clusters']}")
    print("CFD Summary:")
    for key, value in results['CFD']['summary'].items():
        print(f"  {key}: {value}")
    print(f"Detailed CFD: {results['CFD']['detailed'][:10]} ... (truncated)")
    if "Best_Params" in results:
        print(f"Best Parameters: {results['Best_Params']}")



if __name__ == "__main__":
    #Load and preprocess dataset
    file_path = r"D:\\TwoLayerTCP\\data\\tcp.csv"  
    X_train, X_test, y_train, y_test, dynamic_data = load_and_preprocess(file_path)

    
    dynamic_features = dynamic_data[['Duration', 'Cycle', 'LastResults_Success_Percentage', 'LastResults_Length']]
    labels = dynamic_data['Verdict'].values
    state_size = len(dynamic_features)
    action_size = 2

    #Primary Layer Evaluations
    primary_baseline_results = primary_layer_baseline_evaluation(X_train, X_test, y_train, y_test)
    primary_optimized_results = primary_layer_optimized_evaluation(X_train, X_test, y_train, y_test)

    print_results("Primary Layer Baseline", primary_baseline_results)
    print_results("Primary Layer Optimized", primary_optimized_results, include_feature_importances=True)

    #Secondary Layer Evaluations
    secondary_evaluations = {
        "Baseline (K-Means)": kmeans_baseline_evaluation(dynamic_features.copy(), labels),
        "Optimized (K-Means)": kmeans_optimized_evaluation(dynamic_features.copy(), labels),
        "Baseline (Q-Learning)": q_learning_baseline_evaluation(state_size, action_size, dynamic_data, labels, 'Duration'),
        "Optimized (Q-Learning)": q_learning_optimized_evaluation(state_size, action_size, dynamic_data, labels, 'Duration')
    }

    for title, results in secondary_evaluations.items():
        print_secondary_results(title, results)

    #**Framework 1: Hybrid (KNN + SVM) with Optimized K-Means**
    framework_1_results = svm_knn_km_framework(file_path)

    #Print Results for Framework 1
    print("\n--- Framework 1: Hybrid (KNN + SVM) with Optimized K-Means ---")
    print(f"Primary Layer:")
    print(f"  KNN Accuracy: {framework_1_results['Primary Layer']['KNN Accuracy']:.4f}")
    print(f"  SVM Accuracy: {framework_1_results['Primary Layer']['SVM Accuracy']:.4f}")
    print(f"  Hybrid Accuracy: {framework_1_results['Primary Layer']['Hybrid Accuracy']:.4f}")
    print(f"  APFD: {framework_1_results['Primary Layer']['APFD']:.4f}")

    print("\nSecondary Layer:")
    print(f"  Accuracy: {framework_1_results['Secondary Layer']['Accuracy']:.4f}")
    print(f"  Best K: {framework_1_results['Secondary Layer']['Best K']}")
    print(f"  Silhouette Score: {framework_1_results['Secondary Layer']['Silhouette Score']:.4f}")
    
    print(f"\n--- Final Framework 1 Results---")
    print(f"\n Final APFD: {framework_1_results['Final Framework']['APFD']:.4f}")
    print(f" Final Accuracy: {framework_1_results['Final Framework']['Accuracy']:.4f}")
    print(f"\nTop 10 Final Rankings:")
    print(framework_1_results['Final Rankings'])
    
    framework_2_results = knn_svm_q_framework(file_path)

    #Print Framework 2 Results
    print("\n--- Framework 2: Hybrid (KNN + SVM) with Q-Learning ---")
    print("\nPrimary Layer:")
    print(f"  KNN Accuracy: {framework_2_results['Primary Layer']['KNN Accuracy']:.4f}")
    print(f"  SVM Accuracy: {framework_2_results['Primary Layer']['SVM Accuracy']:.4f}")
    print(f"  Hybrid Accuracy: {framework_2_results['Primary Layer']['Hybrid Accuracy']:.4f}")
    print(f"  APFD: {framework_2_results['Primary Layer']['APFD']:.4f}")

    print("\nSecondary Layer:")
    print(f"  Accuracy: {framework_2_results['Secondary Layer']['Accuracy']:.4f}")
    print("  Parameters:")
    print(f"    alpha: {framework_2_results['Secondary Layer']['Parameters']['alpha']}")
    print(f"    gamma: {framework_2_results['Secondary Layer']['Parameters']['gamma']}")
    print(f"    epsilon_decay: {framework_2_results['Secondary Layer']['Parameters']['epsilon_decay']}")

    print("\nFinal Framework 2 Results:")
    print(f"  Accuracy: {framework_2_results['Final Framework']['Accuracy']:.4f}")
    print(f"  APFD: {framework_2_results['Final Framework']['APFD']:.4f}")

    print("\nTop 10 Final Rankings:")
    print(framework_2_results['Final Rankings'])
    
    framework_3_results = moo_km_framework(file_path)

    #Print Results for Framework 3
    print("\n--- Framework 3: Multi-Objective Optimization with Optimized K-Means ---")
    print("\nPrimary Layer:")
    print(f"  KNN Accuracy: {framework_3_results['Primary Layer']['KNN Accuracy']:.4f}")
    print(f"  SVM Accuracy: {framework_3_results['Primary Layer']['SVM Accuracy']:.4f}")
    print(f"  Hybrid Accuracy: {framework_3_results['Primary Layer']['Hybrid Accuracy']:.4f}")
    print(f"  APFD: {framework_3_results['Primary Layer']['APFD']:.4f}")
    print(f"  KNN Best Params: {framework_3_results['Primary Layer']['KNN Best Params']}")
    print(f"  SVM Best Params: {framework_3_results['Primary Layer']['SVM Best Params']}")

    print("\nSecondary Layer:")
    print(f"  Accuracy: {framework_3_results['Secondary Layer']['Accuracy']:.4f}")
    print(f"  Best K: {framework_3_results['Secondary Layer']['Best K']}")
    print(f"  Silhouette Score: {framework_3_results['Secondary Layer']['Silhouette Score']:.4f}")

    print("\nFinal Framework 3 Results:")
    print(f"  Accuracy: {framework_3_results['Final Framework']['Accuracy']:.4f}")
    print(f"  APFD: {framework_3_results['Final Framework']['APFD']:.4f}")

    print("\nTop 10 Final Rankings:")
    print(framework_3_results['Final Rankings'])
    
    framework_4_results = moo_q_framework(file_path)
    
    # Print Results for Framework 4
    print("\n--- Framework 4: Multi-Objective Optimization with Q-learning ---")
    print("\nPrimary Layer:")
    print(f"  KNN Accuracy: {framework_4_results['Primary Layer']['KNN Accuracy']:.4f}")
    print(f"  SVM Accuracy: {framework_4_results['Primary Layer']['SVM Accuracy']:.4f}")
    print(f"  Hybrid Accuracy: {framework_4_results['Primary Layer']['Hybrid Accuracy']:.4f}")
    print(f"  APFD: {framework_4_results['Primary Layer']['APFD']:.4f}")
    print(f"  KNN Best Params: {framework_4_results['Primary Layer']['KNN Best Params']}")
    print(f"  SVM Best Params: {framework_4_results['Primary Layer']['SVM Best Params']}")

    print("\nSecondary Layer:")
    print(f"  Accuracy: {framework_4_results['Secondary Layer']['Accuracy']:.4f}")
    print("  Parameters:")
    print(f"    alpha: {framework_4_results['Secondary Layer']['Parameters']['alpha']}")
    print(f"    gamma: {framework_4_results['Secondary Layer']['Parameters']['gamma']}")
    print(f"    epsilon_decay: {framework_4_results['Secondary Layer']['Parameters']['epsilon_decay']}")

    print("\nFinal Framework 4 Results:")
    print(f"  Accuracy: {framework_4_results['Final Framework']['Accuracy']:.4f}")
    print(f"  APFD: {framework_4_results['Final Framework']['APFD']:.4f}")

    print("\nTop 10 Final Rankings:")
    print(framework_4_results['Final Rankings'])

    