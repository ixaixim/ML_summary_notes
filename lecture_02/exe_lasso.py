### STEPS

# read credit data-set

# split data-set into train and test
    # train data-set: 80%
    # test data-set: 20%


# train model on train data-set ()
    # use sklearn.linear_model.Lasso 
        # For each alpha value in range 0 to 10000: (use logspace)
            # clf = linear_model.Lasso(alpha=0.1)
            # clf.fit(X_train, y_train)

            # predict on test data-set:
            # y_pred = clf.predict(X_test)
            # compute R^2 score: r2_score = clf.score(X_test, y_test)

            # save coefficients and r2_score in a list

# plots

