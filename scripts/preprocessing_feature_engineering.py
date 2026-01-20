import pandas as pd
import numpy as np

def engineer_features(df):
    df['Distance_To_Hydrology'] = np.sqrt(
        df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2
    )
    
    df['Fire_minus_Road'] = df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways']
    
    df['Hillshade_Mean'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']) / 3
    

    
    df['Elevation_adjusted'] = df['Elevation'] - 100 * df['Vertical_Distance_To_Hydrology']
    
    return df

def get_processed_data(file_path, has_labels=True):
    df = pd.read_csv(file_path)
    df = engineer_features(df)
    
    if has_labels:
        if 'Cover_Type' not in df.columns:
            raise ValueError("Cover_Type column not found in data.")
        X = df.drop('Cover_Type', axis=1)
        y = df['Cover_Type']
        return X, y
    else:
        return df, None