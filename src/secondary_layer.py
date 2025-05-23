import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time

# APFD
def evaluate_apfd(prioritized_indices, labels):
    num_test_cases = len(labels)
    num_faults = np.sum(labels)
    if num_faults == 0:
        print("No faults detected in the dataset. APFD cannot be calculated.")
        return None
    fault_positions = [i + 1 for i, idx in enumerate(prioritized_indices) if labels[idx] == 1]
    return 1 - (np.sum(fault_positions) / (num_test_cases * num_faults)) + (1 / (2 * num_test_cases))

#CFD summary
def summarize_cfd(cfd):
    num_cases = len(cfd)
    intervals = [0.25, 0.50, 0.75, 1.00]
    summary = {}
    for interval in intervals:
        index = int(num_cases * interval) - 1
        summary[f"{int(interval * 100)}%"] = round(cfd[index], 2)
    return summary

# Cumulative Fault Detection (CFD)
def calculate_cfd(prioritized_indices, labels):
    cfd = []
    detected_faults = 0
    total_faults = np.sum(labels)
    for i in prioritized_indices:
        if labels[i] == 1:
            detected_faults += 1
        cfd.append(detected_faults / total_faults)
    summary = summarize_cfd(cfd)
    return {"detailed": cfd, "summary": summary}

#Cost per Fault Detected
def calculate_cost_per_fault(total_cost, faults_detected):
    return total_cost / faults_detected if faults_detected > 0 else float('inf')

# K-Means Baseline
def kmeans_baseline_evaluation(dynamic_features, labels):
    start_time = time.time()
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(dynamic_features)
    execution_time = time.time() - start_time

    dynamic_features['Cluster_Baseline'] = clusters
    cluster_ranks = dynamic_features.groupby('Cluster_Baseline')['Duration'].mean().sort_values(ascending=False).index
    prioritized_indices = dynamic_features.sort_values(
        by='Cluster_Baseline',
        key=lambda x: x.map({k: i for i, k in enumerate(cluster_ranks)})
    ).index

    apfd = evaluate_apfd(prioritized_indices, labels)
    cfd = calculate_cfd(prioritized_indices, labels)
    cost_per_fault = calculate_cost_per_fault(execution_time, np.sum(labels))
    ncluster = 2
    return {"APFD": apfd, "CFD": cfd, "Execution_Time": execution_time, "Cost_Per_Fault": cost_per_fault, "Number_of_Clusters": ncluster}

# K-Means Optimized
def kmeans_optimized_evaluation(dynamic_features, labels):
    start_time = time.time()
    best_k = 2
    best_score = float('-inf')
    best_model = None

    for k in range(2, 7):  
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(dynamic_features)
        silhouette_avg = silhouette_score(dynamic_features, clusters)
        if silhouette_avg > best_score:
            best_k = k
            best_score = silhouette_avg
            best_model = kmeans

    clusters = best_model.predict(dynamic_features)
    execution_time = time.time() - start_time

    dynamic_features['Cluster_Optimized'] = clusters
    cluster_ranks = dynamic_features.groupby('Cluster_Optimized')['Duration'].mean().sort_values(ascending=False).index
    prioritized_indices = dynamic_features.sort_values(
        by='Cluster_Optimized',
        key=lambda x: x.map({k: i for i, k in enumerate(cluster_ranks)})
    ).index

    apfd = evaluate_apfd(prioritized_indices, labels)
    cfd = calculate_cfd(prioritized_indices, labels)
    cost_per_fault = calculate_cost_per_fault(execution_time, np.sum(labels))

    return {
        "APFD": apfd,
        "CFD": cfd,
        "Execution_Time": execution_time,
        "Cost_Per_Fault": cost_per_fault,
        "Number_of_Clusters": best_k 
    }

# Q-Learning Baseline 
def q_learning_baseline_evaluation(state_size, action_size, dynamic_data, labels, reward_column):
    class QLearningAgent:
        def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
            self.state_size = state_size
            self.action_size = action_size
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.min_epsilon = min_epsilon
            self.q_table = np.zeros((state_size, action_size))

        def choose_action(self, state):
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.action_size)  
            return np.argmax(self.q_table[state]) 

        def update_q_table(self, state, action, reward, next_state):
            best_next_action = np.argmax(self.q_table[next_state])
            self.q_table[state, action] += self.alpha * (
                reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
            )
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    start_time = time.time()
    agent = QLearningAgent(state_size, action_size)
    for episode in range(200):
        for state in range(state_size):
            action = agent.choose_action(state)
            reward = dynamic_data.iloc[state][reward_column]
            next_state = (state + 1) % state_size
            agent.update_q_table(state, action, reward, next_state)

    execution_time = time.time() - start_time
    prioritized_indices = np.argsort(-agent.q_table[:, 1])
    apfd = evaluate_apfd(prioritized_indices, labels)
    cfd = calculate_cfd(prioritized_indices, labels)
    cost_per_fault = calculate_cost_per_fault(execution_time, np.sum(labels))

    return {"APFD": apfd, "CFD": cfd, "Execution_Time": execution_time, "Cost_Per_Fault": cost_per_fault}

# Q-Learning Optimized 
def q_learning_optimized_evaluation(state_size, action_size, dynamic_data, labels, reward_column):
    class QLearningAgent:
        def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
            self.state_size = state_size
            self.action_size = action_size
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.min_epsilon = min_epsilon
            self.q_table = np.zeros((state_size, action_size))

        def choose_action(self, state):
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.action_size)
            return np.argmax(self.q_table[state])

        def update_q_table(self, state, action, reward, next_state):
            best_next_action = np.argmax(self.q_table[next_state])
            self.q_table[state, action] += self.alpha * (
                reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
            )
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    start_time = time.time()
    best_apfd = float('-inf')
    best_params = None
    best_table = None

    for epsilon_decay in [0.99, 0.995, 0.999]:
        for gamma in [0.9, 0.95, 0.99]:
            agent = QLearningAgent(state_size, action_size, epsilon_decay=epsilon_decay, gamma=gamma)
            for episode in range(500):
                for state in range(state_size):
                    action = agent.choose_action(state)
                    reward = dynamic_data.iloc[state][reward_column]
                    next_state = (state + 1) % state_size
                    agent.update_q_table(state, action, reward, next_state)

            prioritized_indices = np.argsort(-agent.q_table[:, 1])
            apfd = evaluate_apfd(prioritized_indices, labels)
            if apfd > best_apfd:
                best_apfd = apfd
                best_params = (agent.alpha, agent.gamma, agent.epsilon_decay)
                best_table = agent.q_table

    execution_time = time.time() - start_time
    prioritized_indices = np.argsort(-best_table[:, 1])
    cfd = calculate_cfd(prioritized_indices, labels)
    cost_per_fault = calculate_cost_per_fault(execution_time, np.sum(labels))

    return {
        "APFD": best_apfd,
        "CFD": cfd,
        "Execution_Time": execution_time,
        "Cost_Per_Fault": cost_per_fault,
        "Best_Params": best_params,
    }
