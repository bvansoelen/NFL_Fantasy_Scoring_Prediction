import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
import nfl_data_py as nfl
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', None)


# Create another model split by position
def preprocessing(filename):
    df = pd.read_csv(filename)
    # Filter where week is >= 5
    df = df[(df['week'] >= 5) & (df['position'].isin(['QB', 'WR', 'RB', 'TE']))]

    seasons = df['season'].unique()
    actual = df[['player_id', 'full_name', 'posteam', 'position', 'pass_attempt', 'rec_attempt', 'rush_attempt',
                 'pass_completions', 'receptions', 'season',
                 'pass_yards_gained', 'rec_yards_gained', 'rush_yards_gained',
                 'pass_touchdown', 'rec_touchdown', 'rush_touchdown',
                 'pass_two_point_conv', 'rec_two_point_conv', 'rush_two_point_conv',
                 'pass_interception', 'fumble_lost', 'total_fantasy_points', 'total_fantasy_points_exp']]

    data = df.drop(['pass_attempt', 'rec_attempt', 'rush_attempt', 'pass_completions',
                       'receptions', 'pass_yards_gained', 'rec_yards_gained', 'rush_yards_gained',
                       'pass_touchdown', 'rec_touchdown', 'rush_touchdown',
                       'pass_two_point_conv', 'rec_two_point_conv', 'rush_two_point_conv',
                       'pass_interception', 'fumble_lost', 'total_fantasy_points_exp',
                       'game_id', 'full_name', 'posteam', 'opponent'], axis=1)

    data = data.fillna(0)

    # convert dtypes to category
    cat_features = ['offense_rank', 'defense_rank', 'position', 'depth_team']

    # for each of the categorical features, set the datatype to category
    for feature in cat_features:
        dtype = pd.CategoricalDtype(categories=list(set(data[feature])), ordered=False)
        for df in [data]:
            df[feature] = df[feature].astype(dtype)
    return data, actual, seasons


def process(data, actual, seasons):
    predictions = pd.DataFrame()
    feature_importances_df = pd.DataFrame()
    for season in seasons:
        for position in data['position'].unique():

            X_pos = data[(data['position'] == position) & (data['season'] == season)].drop(['total_fantasy_points'], axis=1)
            y_pos = data[(data['position'] == position) & (data['season'] == season)].pop('total_fantasy_points')
            X_train, X_test, y_train, y_test = train_test_split(X_pos, y_pos, test_size=0.2, random_state=42)

            attributes = X_test[['player_id', 'week']]
            X_test = X_test.drop('player_id', axis=1)
            X_train = X_train.drop('player_id', axis=1)

            n_estimators = 1000
            learning_rate = 0.05
            param_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'enable_categorical': True}
            model = XGBRegressor(**param_grid)
            model.fit(X_train, y_train)

            # RMSE Predicted from my model
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            # Combine preds with Ids to see where the model is performing best
            new_preds = pd.DataFrame({'player_id': attributes.player_id, 'week': attributes.week, 'year': season, 'prediction': preds, 'actual': y_test})

            predictions = pd.concat([predictions, new_preds], ignore_index=True)

            # RMSE based on actual fantasy predictions and scores
            actual_rmse = np.sqrt(mean_squared_error(actual[(actual['position'] == position) & (actual['season'] == season)]['total_fantasy_points'],
                                                     actual[(actual['position'] == position) & (actual['season'] == season)]['total_fantasy_points_exp']))

            # Add feature importances to df
            # Extract feature importances
            feature_importances = model.feature_importances_

            # Create a DataFrame for the current feature importances
            current_importances_df = pd.DataFrame(feature_importances, index=X_test.columns, columns=[str(season)])

            # Append the current importances to the feature_importance_df
            if feature_importances_df.empty:
                feature_importances_df = current_importances_df
            else:
                feature_importances_df = pd.concat([feature_importances_df, current_importances_df], axis=1)

            print(f'{season} predicted rmse for {position}  {rmse}')
            print(f'{season} actual rmse for {position} {actual_rmse}')
    return predictions, feature_importances_df


data, actual, seasons = preprocessing('NFL_Fantasy_Scoring_Prediction/2021_to_2023_ff_data_w_fpt_avg_and_team_inj.csv')

# Run model
predicted_points, feature_imp = process(data, actual, seasons)

feature_imp.head
# Which players and positions was I most accurate with
predicted_points['absolute_error'] = abs(predicted_points['prediction'] - predicted_points['actual'])
# predicted_points = predicted_points.dropna()

player_names = pd.read_csv('NFL_Fantasy_Scoring_Prediction/2021_to_2023_ff_data_w_fpt_avg_and_team_inj.csv')
player_names = player_names[(player_names['week'] >= 5) & (player_names['position'].isin(['QB', 'WR', 'RB', 'TE']))]
player_names = player_names[['player_id', 'week', 'full_name', 'position']].drop_duplicates()
#player_names = player_names[['player_id', 'week', 'full_name', 'position', 'total_fantasy_points', 'total_fantasy_points_exp']].drop_duplicates()

# Join predicted points w/ player names and predictions from NFLReadr
predicted_points_test = pd.merge(predicted_points, player_names, on=['player_id'], how='left')

# My worst predictions
third_quartile = predicted_points_test['absolute_error'].quantile(0.75)
worst_predictions = predicted_points_test[predicted_points_test['absolute_error'] > third_quartile]

# Which positions appear in my worst predictions most often
worst_predictions_pct = round(worst_predictions['position'].value_counts(normalize=True) * 100, 2)
position_pct = round(predicted_points_test['position'].value_counts(normalize=True) * 100, 2)

pct_df = pd.DataFrame({
    'predictions_pct': worst_predictions_pct,
    'position_pct': position_pct
})

# Transpose to flip axes
pct_df = pct_df.T
ax = pct_df.plot(kind='bar', stacked=True)
# Rotate x-axis labels to appear parallel to the axis
plt.xticks(rotation=0)
# Add labels inside the bar fills
for container in ax.containers:
    ax.bar_label(container, label_type='center')
plt.title('Percentage of total that a position makes up')
plt.xlabel('Position')
plt.ylabel('Pct')
plt.legend(title='Category / Position Pct', bbox_to_anchor=(1.05, 1), loc='upper left')
# TE's are underrepresented in my worst 25% of predictions while Qbs and Rbs are over represented


# Which positions appear in my best predictions most often
first_quartile = predicted_points_test['absolute_error'].quantile(0.25)
best_predictions = predicted_points_test[predicted_points_test['absolute_error'] < first_quartile]
best_predictions_pct = round(best_predictions['position'].value_counts(normalize=True) * 100, 2)

best_pct_df = pd.DataFrame({
    'predictions_pct': best_predictions_pct,
    'actual_pct': position_pct
})

best_pct_df = best_pct_df.T

ax = best_pct_df.plot(kind='bar', stacked=True)
plt.xticks(rotation=0)

for container in ax.containers:
    ax.bar_label(container, label_type='center')
plt.title('Percentage of total that a position makes up')
plt.xlabel('Position')
plt.ylabel('Pct')
# Similarly TEs are overrepresented in my best predictions while Qbs are underrepresented, and WR and RBs are about even

# How does this compare to actual predictions


abs_desc = predicted_points_test.sort_values(by='absolute_error', ascending=True)
abs_desc.head()

###############################
# EDA
df = pd.read_csv('NFL_Fantasy_Scoring_Prediction/2021_to_2023_ff_data_w_fpt_avg_and_team_inj.csv')
df.describe()
df.isna().sum()

duplicate_rows = df[df.duplicated()]


# na depth
na_depth = df[df['depth_team'].isna()]

na_name = df[df['full_name'].isna()]
na_name['player_id'].head()

# players per year
ppy = df.groupby(['season'])['player_id'].nunique().reset_index()
ppy['player_id'].sum()


df.dtypes

################################
# EDA
# Correlation - Look at correlation between features and target
corr_values = X.drop(cat_features, axis=1)
cc = np.corrcoef(corr_values.values)
plt.figure(figsize=(11, 11))
sns.heatmap(cc*10, center=0, cmap='coolwarm', annot=True, fmt='.0f',
            xticklabels=corr_values.columns, yticklabels=corr_values.columns)
plt.title('Correlation matrix')
plt.show()

# Mutual Information
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=True)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


discrete_features = [col for col in X.columns if col not in cat_features]
x_discrete = X[discrete_features]
mi_scores = make_mi_scores(x_discrete, y, discrete_features)
# print mi_scores
print(mi_scores)
print(mi_scores.tail(3).index)



################################
# First / Original Model
# Create a simple model that works
df = pd.read_csv('2023_ff_data.csv')
# Filter where week is >= 5
df = df[(df['week'] >= 5) & (df['position'].isin(['QB', 'WR', 'RB', 'TE']))]

actual = df[['player_id', 'full_name', 'posteam', 'position', 'pass_attempt','rec_attempt', 'rush_attempt', 'pass_completions', 'receptions',
       'pass_yards_gained', 'rec_yards_gained', 'rush_yards_gained',
       'pass_touchdown', 'rec_touchdown', 'rush_touchdown',
       'pass_two_point_conv', 'rec_two_point_conv', 'rush_two_point_conv',
       'pass_interception', 'rec_fumble_lost', 'total_fantasy_points', 'total_fantasy_points_exp']]

y = df["total_fantasy_points"]

X = df.drop(['pass_attempt', 'rec_attempt', 'rush_attempt', 'pass_completions',
                   'receptions', 'pass_yards_gained', 'rec_yards_gained', 'rush_yards_gained',
                   'pass_touchdown', 'rec_touchdown', 'rush_touchdown',
                   'pass_two_point_conv', 'rec_two_point_conv', 'rush_two_point_conv',
                   'pass_interception', 'rec_fumble_lost', 'total_fantasy_points', 'total_fantasy_points_exp',
                   'season', 'game_id', 'player_id', 'full_name', 'week'], axis=1)

X = X.fillna(0)

# convert dtypes to category
cat_features = ['offense_rank', 'defense_rank', 'opponent', 'posteam', 'position', 'depth_position']

# for each of the categorical features, set the datatype to category
for feature in cat_features:
    dtype = pd.CategoricalDtype(categories=list(set(X[feature])), ordered=False)
    for df in [X]:
        df[feature] = df[feature].astype(dtype)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(enable_categorical=True)
model.fit(X_train, y_train)

xgboost.plot_importance(model, importance_type='weight')
plt.title('Feature Importance')
plt.show()

preds = model.predict(X_test)


pred_vs_actual = pd.DataFrame({'predicted': preds, 'actual': y_test})
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(rmse)


rmse_data = np.sqrt(mean_squared_error(actual['total_fantasy_points'], actual['total_fantasy_points_exp']))
print(rmse_data)

for position in actual['position'].unique():
    actual_rmse = np.sqrt(mean_squared_error(actual[actual['position'] == f'{position}']['total_fantasy_points'],
                                             actual[actual['position'] == f'{position}']['total_fantasy_points_exp']))
    print(f'Actual rmse for {position}: {actual_rmse}')
