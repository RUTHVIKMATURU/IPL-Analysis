import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
cleaned_matches_df = pd.read_csv(r'D:\DE\cleaned_matches.csv')
cleaned_ipl_df = pd.read_csv(r'D:\DE\cleaned_deliveries.csv')
merged_df = pd.merge(cleaned_ipl_df, cleaned_matches_df[['id', 'venue']], left_on='match_id', right_on='id')
def analyze_innings_scores(df):
    innings_scores = df.groupby(['match_id', 'inning', 'batting_team', 'venue'])['total_runs'].sum().reset_index()
    avg_scores = innings_scores.groupby('venue')['total_runs'].mean().sort_values(ascending=False)
    print("Average scores by venue:")
    print(avg_scores)
    return innings_scores
innings_scores = analyze_innings_scores(merged_df)
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
    balls_faced = df[df['extras_type'].isin(['', 'noballs', 'wides']) == False].groupby(['match_id', 'inning']).size().reset_index(name='balls')
    features = features.merge(balls_faced, on=['match_id', 'inning'])
    features['run_rate'] = features['total_runs'] * 6 / features['balls']
    return features
features_df = create_features(merged_df)
X = features_df[['runs_off_bat', 'extras', 'wickets', 'balls', 'run_rate']]
y = features_df['total_runs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
def predict_score(model, venue, batting_team, batting_order):
    venue_avg_score = features_df[features_df['venue'] == venue]['total_runs'].mean()
    avg_runs_off_bat = features_df['runs_off_bat'].mean()
    avg_extras = features_df['extras'].mean()
    avg_wickets = features_df['wickets'].mean()
    avg_balls = features_df['balls'].mean()
    if batting_order == 'batting_first':
        avg_wickets = 10
    elif batting_order == 'batting_second':
        avg_wickets = 0
    input_data = np.array([[avg_runs_off_bat, avg_extras, avg_wickets, avg_balls, venue_avg_score]])
    predicted_score = model.predict(input_data)[0]
    predicted_score += 25
    return predicted_score

import joblib
joblib.dump(rf_model, r'D:\DE\ipl_score_predictor_model.joblib')
print("\nModel saved successfully.")
test_data_df = pd.read_csv(r'D:\DE\test.csv')
for index, row in test_data_df.iterrows():
    match_id = row['match_id']
    venue = row['venue']
    batting_team = row['batting_team']
    batting_order = row['batting_order']
    
    predicted_score = predict_score(rf_model, venue, batting_team, batting_order)
    print(f"\nPredicted score for {batting_team} at {venue} ({batting_order}): {predicted_score:.2f}")
