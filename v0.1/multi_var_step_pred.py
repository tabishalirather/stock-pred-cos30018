'''
Understanding of the task:
1) Prediction should be based on multiple features(I think I already implement this in the previous version)
2) Prediction should be based on multiple time steps. Again, I think I already implement this in the previous version, at least in some form.
3) Actually multi-step prediction means that we are predicting multiple time steps into the future. This is different from predicting a single time step into the future and different from look at multiple steps int the past.
4) Simplest form of multi-variate prediction takes multiple features as input, but we can extend it to take the following as input as well: For examples, the time series of related companies in the same sector. or time series of the market index, or time series of competitors, or time we can have a model that determines a hierarchy of companies and uses the time series of the parent company to predict the time series of the child company in addition to the using the time series of the child company.

1. Implement a function that solve the multistep prediction problem to allow the prediction to be made for a sequence of closing prices of k days into the future.

2. Implement a function that solve the simple multivariate prediction problem to that takes into account the other features for the same company (including opening price, highest price, lowest price, closing price, adjusted closing price, trading volume) as the input for predicting the closing price of the company for a specified day in the future.
'''