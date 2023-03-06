"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
# (you will probably need to import more things here)
import numpy as np
from regression import (logreg, utils, LogisticRegressor)
from math import isclose
from sklearn.preprocessing import StandardScaler
from scipy import stats


def test_prediction():
	
	# Load data
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

    # For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)

	extra_col = np.ones(X_train.shape[0])
	X_train = np.hstack([X_train, extra_col[:, None]])
	preds = log_model.make_prediction(X_train)

	#check that all predictions are between 0 and 1
	assert len(preds[(preds >= 0) & (preds <= 1)]) == len(preds)


def test_loss_function():

	real_y = [1,1,1,0,]
	pred_y = [0.9, 0.8, 0.5, 0.1]

	lr = LogisticRegressor(num_feats=4)
	loss_vals = lr.loss_function(real_y, pred_y)

	#value from sklearn.metrics.logloss (see scratch.ipynb)
	assert isclose(loss_vals, 0.2817529407974519)

def test_gradient():

	# Load data
		X_train, X_val, y_train, y_val = utils.loadDataset(
			features=[
				'Penicillin V Potassium 500 MG',
				'Computed tomography of chest and abdomen',
				'Plain chest X-ray (procedure)',
				'Low Density Lipoprotein Cholesterol',
				'Creatinine',
				'AGE_DIAGNOSIS'
			],
			split_percent=0.8,
			split_seed=42
		)

		# Scale the data, since values vary across feature. Note that we
		# fit on the training data and use the same scaler for X_val.
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_val = sc.transform(X_val)

		# For testing purposes, once you've added your code.
		# CAUTION: hyperparameters have not been optimized.
		log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)

		extra_col = np.ones(X_train.shape[0])
		X_train = np.hstack([X_train, extra_col[:, None]])
		preds = log_model.make_prediction(X_train)

		grad = log_model.calculate_gradient(y_train, X_train,)

		#check that gradient shape equals the number of features
		assert grad.shape[0] == log_model.num_feats + 1


def test_training():

	# Load data
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=13
    )

    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

    # For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.

	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=100, batch_size=100)
	log_model.train_model(X_train,  y_train, X_val, y_val)

	loss_vals = log_model.loss_hist_train
	iters = np.arange(len(loss_vals))

	#check that in general, loss goes down over time
	slope, intercept, r_value, p_value, std_err = stats.linregress(iters,loss_vals)
	assert slope < 0