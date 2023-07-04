# STEPS

# read nitride_compounds.csv data-set (only atom properties)

# split data-set into train and test

# For fraction of training data
# train model with n-fold cross validation on train data-set ()
    # define estimator: kernel_ridge.KernelRidge(alpha=1.0)
    # define hyper-parameter space: param_grid = {"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}
    # define grid-search: grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='mean_squared_error')
    # fit grid-search: grid.fit(X_train, y_train)
    # predict on test data-set: y_pred = grid.predict(X_test)
    # NOTE: do not use test data, but separate again into train and test data-set within the cross validation

# plot best model performance 