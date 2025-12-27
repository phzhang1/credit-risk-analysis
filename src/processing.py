import pandas as pd

def load_clean_data(filepath):
    """
    Loads, cleans, and encodes the Credit Risk dataset for modeling 
    through this preprocessing pipeline:

    1. Load the Data from CSV.
    2. Remove realistic outliers (Age > 100, Employment > 60 years).
    3. Impute missing values using segmentation imputation.
    4. Encodes categorical variables (Ordinal for Grades, One-Hot for others).

    Args:
        filepath (str): The file path to the raw CSV dataset.
    
    Returns:
        pd.DataFrame: The fully processed dataframe, ready for training.
    """

    # Load Data
    df = pd.read_csv(filepath)

    # Handle Outliers
    # We remove ages > 100 and employment > 60 as these are unrealistic and likely entry errors.
    df = df[df['person_age'] < 100]
    df = df[df['person_emp_length'] < 60]

    # Impute Missing Values
    # Fill missing Interest Rates with the median rate of their specific Loan Grade
    df['loan_int_rate'] = df['loan_int_rate'].fillna(
        df.groupby('loan_grade')['loan_int_rate'].transform('median')
    )

    # Fill missing Employment Length with the median of their Home Ownership status
    df['person_emp_length'] = df['person_emp_length'].fillna(
        df.groupby('person_home_ownership')['person_emp_length'].transform('median')
    )

    # Encode Categorical Variables
    # Ordinal Encoding for Loan Grade (A=0 is best, G=6 is the worst)
    grade_map = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6}
    df['loan_grade'] = df['loan_grade'].map(grade_map)

    # Binary Encoding 
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

    # One-Hot Encoding for nominal categories
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

    # Ensure all boolean columns are converted to integers for machine readability.
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)

    return df

