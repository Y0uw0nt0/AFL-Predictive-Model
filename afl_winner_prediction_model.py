
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

#loads data
player_stats = pd.read_csv('data/afl_player_stats.csv')
match_results = pd.read_csv('data/afl_match_results.csv')
fixture_2018 = pd.read_csv('data/afl_fixture_2018.csv')

#filters seasons with consistent stats
player_stats = player_stats[player_stats['Season'].between(2015, 2018)].copy()

#create MatchID
player_stats['MatchID'] = (
    player_stats['Season'].astype(str) + '_' +
    player_stats['Team'] + '_' +
    player_stats['Venue'] + '_' +
    player_stats['Status']
)

#aggregates to team level
team_stats = player_stats.groupby(['Season', 'Team', 'Venue', 'Status', 'MatchID']).mean(numeric_only=True).reset_index()

#preps training data 
train_stats = team_stats[team_stats['Season'] < 2018] #change if dataset is expanded
match_results_train = match_results[match_results['Season'].between(2015, 2017)].copy()
match_results_train['HomeMatchID'] = (
    match_results_train['Season'].astype(str) + '_' +
    match_results_train['Home.Team'] + '_' +
    match_results_train['Venue'] + '_Home'
)
match_results_train['AwayMatchID'] = (
    match_results_train['Season'].astype(str) + '_' +
    match_results_train['Away.Team'] + '_' +
    match_results_train['Venue'] + '_Away'
)

home_stats = train_stats.rename(columns=lambda x: f'Home_{x}' if x not in ['MatchID'] else x)
away_stats = train_stats.rename(columns=lambda x: f'Away_{x}' if x not in ['MatchID'] else x)

matches_train = match_results_train.merge(home_stats, left_on='HomeMatchID', right_on='MatchID', how='inner')
matches_train = matches_train.merge(away_stats, left_on='AwayMatchID', right_on='MatchID', how='inner')
matches_train['Home_Win'] = (matches_train['Home.Points'] > matches_train['Away.Points']).astype(int)

#preps features and labels
X = matches_train.select_dtypes(include='number').drop(columns=['Home.Points', 'Away.Points', 'Home_Win'])
y = matches_train['Home_Win']

#imputes missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

#trains model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_imputed, y)

#preps results data
fixture_2018['HomeMatchID'] = (
    fixture_2018['Season'].astype(str) + '_' +
    fixture_2018['Home.Team'] + '_' +
    fixture_2018['Venue'] + '_Home'
)
fixture_2018['AwayMatchID'] = (
    fixture_2018['Season'].astype(str) + '_' +
    fixture_2018['Away.Team'] + '_' +
    fixture_2018['Venue'] + '_Away'
)

team_stats_2018 = team_stats[team_stats['Season'] == 2018]
home_stats_2018 = team_stats_2018.rename(columns=lambda x: f'Home_{x}' if x not in ['MatchID'] else x)
away_stats_2018 = team_stats_2018.rename(columns=lambda x: f'Away_{x}' if x not in ['MatchID'] else x)

matches_2018 = fixture_2018.merge(home_stats_2018, left_on='HomeMatchID', right_on='MatchID', how='inner')
matches_2018 = matches_2018.merge(away_stats_2018, left_on='AwayMatchID', right_on='MatchID', how='inner')

#predict results
trained_features = pd.DataFrame(X).columns
X_2018 = matches_2018[trained_features.intersection(matches_2018.columns)]
X_2018 = X_2018.reindex(columns=trained_features)
X_2018_imputed = imputer.transform(X_2018)
y_2018_pred = model.predict(X_2018_imputed)

#compare predictions with results
match_results_2018 = match_results[match_results['Season'] == 2018].copy()
match_results_2018['HomeMatchID'] = (
    match_results_2018['Season'].astype(str) + '_' +
    match_results_2018['Home.Team'] + '_' +
    match_results_2018['Venue'] + '_Home'
)
match_results_2018['AwayMatchID'] = (
    match_results_2018['Season'].astype(str) + '_' +
    match_results_2018['Away.Team'] + '_' +
    match_results_2018['Venue'] + '_Away'
)

predicted_ids = matches_2018[['HomeMatchID', 'AwayMatchID']].copy()
predicted_ids['Pred_HomeWin'] = y_2018_pred

results_comparison = predicted_ids.merge(
    match_results_2018[['HomeMatchID', 'AwayMatchID', 'Home.Points', 'Away.Points']],
    on=['HomeMatchID', 'AwayMatchID'],
    how='inner'
)
results_comparison['Actual_HomeWin'] = (results_comparison['Home.Points'] > results_comparison['Away.Points']).astype(int)

#tell how accurate
accuracy = accuracy_score(results_comparison['Actual_HomeWin'], results_comparison['Pred_HomeWin'])
print("2018 Prediction Accuracy:", accuracy)
