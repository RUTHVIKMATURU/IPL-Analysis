import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
try:
    matches_df = pd.read_csv(r'D:\DE\matches.csv')
    ipl_df = pd.read_csv(r'D:\DE\deliveries.csv')
    print("Columns in deliveries.csv:", ipl_df.columns.tolist())
except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure the CSV files exist in the specified location.")
    exit(1)

# Cleaning matches.csv
def clean_matches_data(df):
    # Convert date to datetime, handling potential errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Fill NaN dates with a placeholder date
    df['date'] = df['date'].fillna(pd.Timestamp('1900-01-01'))
    
    # Fill NaN values in result_margin with 0
    df['result_margin'] = df['result_margin'].fillna(0)
    
    # Convert result_margin to integer
    df['result_margin'] = df['result_margin'].astype(int)
    
    # Fill NaN values in city with 'Unknown'
    df['city'] = df['city'].fillna('Unknown')
    
    # Fill NaN values in player_of_match with 'Not Awarded'
    df['player_of_match'] = df['player_of_match'].fillna('Not Awarded')
    
    # Fill NaN values in venue with 'Unknown'
    df['venue'] = df['venue'].fillna('Unknown')
    
    # Convert target_runs and target_overs to float
    df['target_runs'] = df['target_runs'].fillna(0).astype(float)
    df['target_overs'] = df['target_overs'].fillna(0).astype(float)
    
    # Convert super_over to boolean
    df['super_over'] = df['super_over'].map({'N': False, 'Y': True})
    
    return df

# Cleaning deliveries.csv
def clean_ipl_data(df, matches_df):
    # Merge with matches_df to get the date
    df = df.merge(matches_df[['id', 'date']], left_on='match_id', right_on='id', how='left')
    df.drop('id', axis=1, inplace=True)
    
    # Convert date to datetime, handling potential errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Fill NaN dates with a placeholder date
    df['date'] = df['date'].fillna(pd.Timestamp('1900-01-01'))
    
    # Fill NaN values in batting_team and bowling_team with 'Unknown'
    df['batting_team'] = df['batting_team'].fillna('Unknown')
    df['bowling_team'] = df['bowling_team'].fillna('Unknown')
    
    # Fill NaN values in batsman and non_striker with 'Unknown'
    df['batter'] = df['batter'].fillna('Unknown')
    df['non_striker'] = df['non_striker'].fillna('Unknown')
    
    # Fill NaN values in bowler with 'Unknown'
    df['bowler'] = df['bowler'].fillna('Unknown')
    
    # Convert numeric columns to appropriate types
    numeric_columns = ['inning', 'over', 'ball', 'total_runs', 'batsman_runs', 'extra_runs', 'wide_runs', 'noball_runs', 'bye_runs', 'legbye_runs', 'penalty_runs']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Convert boolean columns if they exist
    boolean_columns = ['is_super_over']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].map({0: False, 1: True})
    
    return df

# Clean the datasets
cleaned_matches_df = clean_matches_data(matches_df)
cleaned_ipl_df = clean_ipl_data(ipl_df, matches_df)

# Save the cleaned datasets
output_dir = r'D:\DE'
try:
    cleaned_matches_df.to_csv(os.path.join(output_dir, 'cleaned_matches.csv'), index=False)
    cleaned_ipl_df.to_csv(os.path.join(output_dir, 'cleaned_deliveries.csv'), index=False)
    print("Data cleaning completed. Cleaned datasets saved.")
except PermissionError:
    print("Error: Unable to save files. Please check if you have write permissions to the output directory.")
except Exception as e:
    print(f"An unexpected error occurred while saving the files: {e}")

# Load the cleaned datasets
cleaned_matches_df = pd.read_csv(r'D:\DE\cleaned_matches.csv')
cleaned_ipl_df = pd.read_csv(r'D:\DE\cleaned_deliveries.csv')

# Merge the datasets
merged_df = pd.merge(cleaned_ipl_df, cleaned_matches_df[['id', 'venue']], left_on='match_id', right_on='id')

# Perform innings score analysis
def analyze_innings_scores(df):
    innings_scores = df.groupby(['match_id', 'inning', 'batting_team', 'venue'])['total_runs'].sum().reset_index()
    avg_scores = innings_scores.groupby('venue')['total_runs'].mean().sort_values(ascending=False)
    print("Average scores by venue:")
    print(avg_scores)
    return innings_scores

innings_scores = analyze_innings_scores(merged_df)

# Create features for prediction
def create_features(df):
    features = df.groupby(['match_id', 'inning', 'batting_team', 'venue']).agg({
        'total_runs': 'sum',
        'batsman_runs': 'sum',
        'extra_runs': 'sum',
        'is_wicket': 'sum'
    }).reset_index()

    features = features.rename(columns={
        'is_wicket': 'wickets',
        'batsman_runs': 'runs_off_bat',
        'extra_runs': 'extras'
    })

    # Calculate the total number of balls faced (excluding extras)
    balls_faced = df[df['extras_type'].isin(['', 'noballs', 'wides']) == False].groupby(['match_id', 'inning']).size().reset_index(name='balls')
    features = features.merge(balls_faced, on=['match_id', 'inning'])

    # Calculate run rate
    features['run_rate'] = features['total_runs'] * 6 / features['balls']

    return features

features_df = create_features(merged_df)

# Prepare data for machine learning
X = features_df[['runs_off_bat', 'extras', 'wickets', 'balls', 'run_rate']]
y = features_df['total_runs']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Function to predict score for a particular match
def predict_score(model, venue, batting_team, batting_order):
    venue_avg_score = features_df[features_df['venue'] == venue]['total_runs'].mean()
    
    # Use average values for other features
    avg_runs_off_bat = features_df['runs_off_bat'].mean()
    avg_extras = features_df['extras'].mean()
    avg_wickets = features_df['wickets'].mean()
    avg_balls = features_df['balls'].mean()
    
    # Adjust the average values based on the batting order
    if batting_order == 'batting_first':
        avg_wickets = 10  # Assuming a team has 10 wickets in an innings
    elif batting_order == 'batting_second':
        avg_wickets = 0  # Assuming no wickets in an innings
    
    input_data = np.array([[avg_runs_off_bat, avg_extras, avg_wickets, avg_balls, venue_avg_score]])
    predicted_score = model.predict(input_data)[0]
    
    # Add 25 to the predicted score
    predicted_score += 25
    
    return predicted_score

# Example usage
venue = "M Chinnaswamy Stadium"
batting_team = "Royal Challengers Bangalore"

# Predicting score for batting first
predicted_score_batting_first = predict_score(rf_model, venue, batting_team, 'batting_first')
print(f"\nPredicted score for {batting_team} at {venue} (batting first): {predicted_score_batting_first:.2f}")

# Predicting score for batting second
predicted_score_batting_second = predict_score(rf_model, venue, batting_team, 'batting_second')
print(f"\nPredicted score for {batting_team} at {venue} (batting second): {predicted_score_batting_second:.2f}")

# Save the model for future use
import joblib
joblib.dump(rf_model, r'D:\DE\ipl_score_predictor_model.joblib')
print("\nModel saved successfully.")

# Load the test data
test_data_df = pd.read_csv(r'D:\DE\test.csv')

# Predict scores for the test data
for index, row in test_data_df.iterrows():
    match_id = row['match_id']
    venue = row['venue']
    batting_team = row['batting_team']
    batting_order = row['batting_order']
    
    predicted_score = predict_score(rf_model, venue, batting_team, batting_order)
    print(f"\nPredicted score for {batting_team} at {venue} ({batting_order}): {predicted_score:.2f}")
