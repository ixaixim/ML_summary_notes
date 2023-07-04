# STEPS

# read wave.csv data-set

# split data-set into train and test

# train model on train data-set ()
    # define estimator: kernel_ridge.KernelRidge(alpha=1.0)
    # define hyper-parameter space: param_grid = {"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}
    # define grid-search: grid = GridSearchCV(estimator=estimator, param_grid=param_grid)
    # fit grid-search: grid.fit(X_train, y_train)
    # predict on test data-set: y_pred = grid.predict(X_test)
    # compute MSE: mse = mean_squared_error(y_test, y_pred)
    # save coefficients if mse is minimum

# plots: indicate MSE, MAE, R^2 score in title
# reasoning: 
# short argument how you think, the model could be improved, provided you could evaluate arbitrary points as input for your machine learning procedure (but not more, than given to you in this case) - also think about the apparent limitationsof the model for predicting the function.
# E.g. more points in the high oscillating regions would be helpful. 
# store in a file args. txt 

