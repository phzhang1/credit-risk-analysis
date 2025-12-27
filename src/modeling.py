from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression

def train_lr(X_train, y_train):
    """
    Trains a Logistic Regression model to serve as a baseline.

    Uses 'liblinear' for smaller datasets and binary classification.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target labels.

        Returns:
            LogisticRegression: The fitted Logistic Regression model.
    """

    lr_model = LogisticRegression(
        solver='liblinear',
        random_state=7,
        max_iter=1000
    )

    lr_model.fit(X_train, y_train)

    return lr_model

def train_rf(X_train, y_train):
    """
    Trains a Random Forest Classifier with balanced class weights.

    This model is intended for the primary supervised learning model 
    for predicting loan defaults. It uses 'balanced' class weights 
    to handle the imbalance in the target variable.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target labels (0 = Safe, 1 = Default).

        Returns:
            RandomForestClassifier: The fitted Random Forest model.
    """
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=7,
        class_weight='balanced'
    )

    rf_model.fit(X_train, y_train)

    return rf_model

def train_if(X_train):
    """
    Trains an Isolation Forest to detect anomalies (Hidden Risks).
    
    This unsupervised model identifies statistical outliers (potential fraud)
    that deviate from normal financial behavior. It assumes a contamination
    rate of 5% to flag unusual profiles.

    Args:
        X_train (pd.DataFrame): The training features.

        Returns:
            IsolationForest: The fitted Isolation Forest model.
    """
    if_model = IsolationForest(
        contamination=0.05,
        random_state=7
    )

    if_model.fit(X_train)

    return if_model