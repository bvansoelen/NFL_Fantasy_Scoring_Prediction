import pandas as pd
import numpy as np
import matplotlib as plt
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

    actual = df[['player_id', 'full_name', 'posteam', 'position', 'pass_attempt', 'rec_attempt', 'rush_attempt',
                 'pass_completions', 'receptions',
                 'pass_yards_gained', 'rec_yards_gained', 'rush_yards_gained',
                 'pass_touchdown', 'rec_touchdown', 'rush_touchdown',
                 'pass_two_point_conv', 'rec_two_point_conv', 'rush_two_point_conv',
                 'pass_interception', 'fumble_lost', 'total_fantasy_points', 'total_fantasy_points_exp']]

    data = df.drop(['pass_attempt', 'rec_attempt', 'rush_attempt', 'pass_completions',
                       'receptions', 'pass_yards_gained', 'rec_yards_gained', 'rush_yards_gained',
                       'pass_touchdown', 'rec_touchdown', 'rush_touchdown',
                       'pass_two_point_conv', 'rec_two_point_conv', 'rush_two_point_conv',
                       'pass_interception', 'fumble_lost', 'total_fantasy_points_exp',
                       'season', 'game_id', 'player_id', 'full_name', 'week', 'posteam', 'opponent'], axis=1)

    data = data.fillna(0)

    # convert dtypes to category
    cat_features = ['offense_rank', 'defense_rank', 'position', 'depth_team']

    # for each of the categorical features, set the datatype to category
    for feature in cat_features:
        dtype = pd.CategoricalDtype(categories=list(set(data[feature])), ordered=False)
        for df in [data]:
            df[feature] = df[feature].astype(dtype)
    return data, actual


def process(data, actual):
    for position in data['position'].unique():
        X_pos = data[data['position'] == position].drop('total_fantasy_points', axis=1)
        y_pos = data[data['position'] == position].pop('total_fantasy_points')
        X_train, X_test, y_train, y_test = train_test_split(X_pos, y_pos, test_size=0.2, random_state=42)

        n_estimators = 1000
        learning_rate = 0.05
        param_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'enable_categorical': True}
        model = XGBRegressor(**param_grid)
        model.fit(X_train, y_train)

        # RMSE Predicted from my model
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # RMSE based on actual fantasy predictions and scores
        actual_rmse = np.sqrt(mean_squared_error(actual[actual['position'] == position]['total_fantasy_points'],
                                                 actual[actual['position'] == position]['total_fantasy_points_exp']))

        print(f'Predicted rmse for {position}  {rmse}')
        print(f'Actual rmse for {position} {actual_rmse}')


data, actual = preprocessing('2021_to_2023_ff_data_w_fpt_avg_and_team_inj.csv')

# Run model
process(data, actual)

###############################
# EDA
df = pd.read_csv('2021_to_2023_ff_data_w_fpt_avg_and_team_inj.csv')
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
