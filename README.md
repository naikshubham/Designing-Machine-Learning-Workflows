# Designing-Machine-Learning-Workflows
Deploying machine learning models in production seems easy with modern tools, but often ends in disappointment as the model performs worse in production than in development. How to exhaustively tune every aspect of our model in development; how to make the best possible use of available domain expertise; how to monitor our model in performance and deal with any performance deterioration; and finally how to deal with poorly or scarcely labelled data. Digging deep into the cutting edge of sklearn, and dealing with real-life datasets from hot areas like personalized healthcare and cybersecurity.


### Feature selection

- Create fake variables and augment dataset

```python
from np.random import uniform
fakes = pd.DataFrame(uniform(low=0.0, high=1.0, size=n * 100).reshape(X.shape[0], 100), columns=['fake_' + str(j) for j in range(100)])

X_with_fakes = pd.concat([X, fakes], 1)
```

- We use the `SelectKBest` algorithm from the `feature_selector` module to select the 20 highest-scoring columns. We use the `chi2` scoring method. The feature selector has a `fit()` method to fit it to the data,and a `get_support()` method that returns the index of the selected columns.
