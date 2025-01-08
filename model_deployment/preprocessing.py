import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(data):
    # Convert input data to DataFrame
    data = pd.DataFrame(data)
    
    categorical_columns=data.select_dtypes(['object']).columns
    # print(categorical_columns)

    # Initialize LabelEncoders for categorical columns
    # categorical_columns = ['home_ownership', 'verification_status', 'application_type', 'city']
    
    # Fit label encoders and transform categorical columns
    for col in categorical_columns:
        if col in data.columns:
            label_encoder = LabelEncoder()
            data[col] = label_encoder.fit_transform(data[col].astype(str))
    
    # # Handle missing values if necessary
    data.fillna(0, inplace=True)
    
    return data
