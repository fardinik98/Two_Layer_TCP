import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def load_and_preprocess(file_path):
    data = pd.read_csv(file_path, delimiter=';')

    # Normalize numeric columns using MinMaxScaler
    def normalize_data(df, columns):
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    # Encode categorical columns
    def encode_categorical(df, columns):
        encoder = LabelEncoder()
        for col in columns:
            df[col] = encoder.fit_transform(df[col])
        return df

    # Process LastResults column
    def process_last_results(column):
        return pd.DataFrame({
            "LastResults_Length": column.apply(lambda x: len(eval(x))),
            "LastResults_Mean": column.apply(
                lambda x: sum(eval(x)) / len(eval(x)) if len(eval(x)) > 0 else 0
            ),
            "LastResults_Fail_Percentage": column.apply(
                lambda x: (eval(x).count(0) / len(eval(x)) * 100) if len(eval(x)) > 0 else 0
            ),
            "LastResults_Success_Percentage": column.apply(
                lambda x: (eval(x).count(1) / len(eval(x)) * 100) if len(eval(x)) > 0 else 0
            ),
        })

    # Process LastResults and add features 
    last_results_features = process_last_results(data["LastResults"])
    data = pd.concat([data, last_results_features], axis=1)

    # Define static and dynamic 
    static_features = ['Duration', 'CalcPrio', 'LastResults_Mean', 'LastResults_Length']
    dynamic_features = ['Duration', 'Cycle', 'LastResults_Success_Percentage', 'LastResults_Length']
    target_column = 'Verdict'

    # Normalize numeric columns
    numeric_columns = ['Duration', 'Cycle']
    data = normalize_data(data, numeric_columns)

    # Apply Winsorization to Verdict
    #def winsorize(column, lower_percentile=1, upper_percentile=99):
        #lower_bound = column.quantile(lower_percentile / 100.0)
        #upper_bound = column.quantile(upper_percentile / 100.0)
        #return column.clip(lower=lower_bound, upper=upper_bound)

    #data[target_column] = winsorize(data[target_column])

    # Encode categorical columns
    data = encode_categorical(data, [target_column])

    # Split data into static and dynamic datasets with stratified sampling
    X_static = data[static_features]
    y_static = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X_static, y_static, test_size=0.3, random_state=42, stratify=y_static)

    # Apply SMOTE to the training set
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # dynamic dataset
    dynamic_data = data[dynamic_features + [target_column]]

    return X_train, X_test, y_train, y_test, dynamic_data
