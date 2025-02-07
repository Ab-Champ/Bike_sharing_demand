!pip install -U pip
!pip install -U setuptools wheel
!pip install -U "mxnet<2.0.0" bokeh==2.0.1
!pip install autogluon --no-cache-dir
# Without --no-cache-dir, smaller aws instances may have trouble installing

!sudo mkdir -p /root/.kaggle
!sudo touch /root/.kaggle/kaggle.json
!sudo chmod 600 /root/.kaggle/kaggle.json

import json
kaggle_username = "xxxxxx"
kaggle_key = "xxxxxx"

# Save API token the kaggle.json file
with open('/.kaggle/kaggle.json', "w") as f:
    f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))
!pip install kaggle
!chmod 600 /home/sagemaker-user/.kaggle/kaggle.json
!kaggle competitions download -c bike-sharing-demand
!unzip -o bike-sharing-demand.zip
import pandas as pd
from autogluon.tabular import TabularPredictor

train = pd.read_csv('train.csv', parse_dates=['datetime'])
train.head()
train.describe()

test = pd.read_csv(r'/home/sagemaker-user/cd0385-project-starter/project/test.csv', parse_dates=['datetime'])
test.head()

submission = pd.read_csv(r'/home/sagemaker-user/cd0385-project-starter/project/sampleSubmission.csv', parse_dates=['datetime'])
submission.head()
predictor = TabularPredictor(label='count', eval_metric='root_mean_squared_error').fit(train.drop(columns=['casual' , 'registered'], axis=1), time_limit=600, presets='best_quality', auto_stack=False)
### Review AutoGluon's training run with ranking of models that did the best.
predictor.fit_summary()
predictions = predictor.predict(test)
predictions.head()
predictions.describe()
No predictions found to be negative
submission["count"] = predictions
submission.to_csv("submission.csv", index=False)
!kaggle competitions submit -c bike-sharing-demand -f submission.csv -m "first raw submission"
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6

# Create a histogram of all features 
train.hist(figsize=(12,12))

# create a new feature
train['hour'] = train['datetime'].dt.hour
test['hour'] = test['datetime'].dt.hour

train['day'] = train['datetime'].dt.day
test['day'] = test['datetime'].dt.day

train['month'] = train['datetime'].dt.month
test['month'] = test['datetime'].dt.month

train['year'] = train['datetime'].dt.year
test['year'] = test['datetime'].dt.year
train["season"] = train["season"].astype('category')
train["weather"] = train["weather"].astype('category')

test["season"] = test["season"].astype('category')
test["weather"] = test["weather"].astype('category')
# View new feature
train.head()
# View histogram of all features 
train.hist(figsize=(10,12))
predictor_new_features = TabularPredictor(label='count', eval_metric='root_mean_squared_error').fit(train.drop(columns=['casual' , 'registered'], axis=1), time_limit=600, presets='best_quality', auto_stack=False)
predictor_new_features.fit_summary()

predictions_new_features=predictor_new_features.predict(test)
predictions_new_features.describe()
predictions_new_features[predictions_new_features < 0] = 0

# Rechecking if any predictions are less than 0
negative_pred_count = predictions_new_features.apply(lambda x: 1 if x < 0 else 0)
pred_neg_count = (negative_pred_count == 1).sum()

# Output the count of negative predictions
print(f"No. of negative predictions: {pred_neg_count}")

# Confirm that all negative values in the predictions have been set to zero
print("All negative values in the predictions (if any) are set to zero successfully.")

predictions_new_features.describe()
All values set to zero
submission_new_features = pd.read_csv('sampleSubmission.csv', parse_dates = ['datetime'])
submission_new_features.head()
# Same submitting predictions
submission_new_features["count"] = predictions_new_features
submission_new_features.to_csv("submission_new_features.csv", index=False)
!kaggle competitions submit -c bike-sharing-demand -f submission_new_features.csv -m "new features"
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
from autogluon.tabular import TabularPredictor
import autogluon.core as ag

# Define hyperparameter options for different models
gbm_options = {
    'extra_trees': True,
    'num_boost_round': ag.space.Int(lower=100, upper=800, default=100),
    'num_leaves': ag.space.Int(lower=26, upper=66, default=36),
    'ag_args': {'name_suffix': 'XT'}
}

xgb_options = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': ag.space.Int(lower=5, upper=8, default=6),
    'n_estimators': ag.space.Int(lower=100, upper=500, default=100),
    'eta': 0.3,
    'subsample': 1,
    'colsample_bytree': 1
}

rf_options = {
    'criterion': 'squared_error',
    'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}
}

# Define hyperparameters dictionary for different model types
hyperparameters = {
    'GBM': gbm_options,
    'XGB': xgb_options,
    'RF': rf_options
}

# Define hyperparameter tuning settings
time_limit = 600
num_trials = 20
search_strategy = 'auto'

hyperparameter_tune_kwargs = {
    'num_trials': num_trials,
    'scheduler': 'local',
    'searcher': search_strategy
}

# Fit TabularPredictor with hyperparameter tuning
predictor_new_hpo = TabularPredictor(label='count', problem_type='regression', eval_metric='root_mean_squared_error').fit(
    train.drop(columns=['casual', 'registered'], axis=1),
    time_limit=time_limit,
    presets='best_quality',
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    auto_stack=False
)

predictor_new_hpo.fit_summary()
# Remember to set all negative values to zero
predictions_new_hpo=predictor_new_hpo.predict(test)
predictions_new_hpo.describe()
# Remember to set all negative values to zero
predictions_new_hpo[predictions_new_hpo < 0] = 0

# Rechecking if any predictions are less than 0
negative_pred_count_hpo = predictions_new_hpo.apply(lambda x: 1 if x < 0 else 0)
pred_neg_count_hpo = (negative_pred_count_hpo == 1).sum()

# Output the count of negative predictions
print(f"No. of negative predictions: {pred_neg_count_hpo}")

# Confirm that all negative values in the predictions have been set to zero
print("All negative values in the predictions (if any) are set to zero successfully.")

predictions_new_hpo.describe()
submission_new_hpo = pd.read_csv('sampleSubmission.csv', parse_dates = ['datetime'])
submission_new_hpo.head()

# Same submitting predictions
submission_new_hpo["count"] = predictions_new_hpo
submission_new_hpo.to_csv("submission_new_hpo.csv", index=False)
!kaggle competitions submit -c bike-sharing-demand -f submission_new_hpo.csv -m "new features with hyperparameters"
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6

# Taking the top model score from each training run and creating a line plot to show improvement
fig = pd.DataFrame(
    {
        "model": ["initial", "add_features", "hpo"],
        "score": [92.442085, 32.234499 ,34.719962 ]
    }
).plot(x="model", y="score", figsize=(8, 6)).get_figure()
fig.savefig('model_train_score.png')

# Take the 3 kaggle scores and creating a line plot to show improvement
fig = pd.DataFrame(
    {
        "test_eval": ["initial", "add_features", "hpo"],
        "score": [1.86412,0.51296 , 0.5198]
    }
).plot(x="test_eval", y="score", figsize=(8, 6)).get_figure()
fig.savefig('model_test_score.png')

# The 3 hyperparameters we tuned with the kaggle score as the result
pd.DataFrame({
    "model": ["initial", "add_features", "hpo"],
    "hpo1": ["default_value", "default_value", "GBM: {(num_boost_round: lower=100, upper=800), (num_leave: lower=26, upper=66)}"],
    "hpo2": ["default_value", "default_value", "XGB: {(max_depth: lower=5, upper=8), (n_estimators: lower=100, upper=500), (eta=0.3), (subsample: 1), (colsample_bytree:1)"],
    "hpo3": ["default_value", "default_value", "rf: (criterion: squared_error) "],
    "score": [1.86412, 0.51296, 0.5198]
})