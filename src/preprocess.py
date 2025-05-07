def preprocess_data(df):
    return train_test_split(X, y, test_size=0.2, random_state=42)