from sklearn.linear_model import Ridge
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# Create Data with a randomized function
m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)
plt.scatter(x, y)


# Ridge Regression (L2 Regularization)
ridge_reg = Ridge(alpha=2, solver='auto', max_iter=1000, tol=0.001)
ridge_reg.fit(x, y)
x_ridge = [[-2]]
y_ridge = ridge_reg.predict(x_ridge)
print(f"Ridge Regression Y value: {y_ridge}")
plt.scatter(x_ridge, y_ridge, c='orange', s=100, label="Ridge Regression")

# Stochastic Gradient Descent with l2 penalty (alternative to above)
sgd_reg = SGDRegressor(penalty='l2', max_iter=1000, tol=0.001)
sgd_reg.fit(x, y.ravel())
x_sgd = [[-1]]
y_sgd = sgd_reg.predict(x_sgd)
print(f"SGD Ridge Y value: {y_sgd}")
plt.scatter(x_sgd, y_sgd, c="red", s=100, label="SGD_l2")


# Lasso Regression (L1 Regularization)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x, y)
x_lasso = [[0]]
y_lasso = lasso_reg.predict(x_lasso)
print(f"Lasso Y value: {y_lasso}")
plt.scatter(x_lasso, y_lasso, c="green", s=100, label="Lasso Regression")

# Stochastic Gradient Descent with l1 penalty (alternative to above)
sgd_lasso = SGDRegressor(penalty='l1', max_iter=100, tol=0.001)
sgd_lasso.fit(x, y.ravel())
x_sgd_lasso = [[1.0]]
y_sgd_lasso = sgd_lasso.predict(x_sgd_lasso)
print(f"SGD Lasso Y value: {y_sgd_lasso}")
plt.scatter(x_sgd_lasso, y_sgd_lasso, c="magenta", s=100, label="SGD_l1")


# Elastic Net - uses a mix of l1 and l2 regularization
elastic_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_reg.fit(x, y)
x_elastic = [[2.0]]
y_elastic = elastic_reg.predict(x_elastic)
print(f"Elastic Net Y value: {y_elastic}")
plt.scatter(x_elastic, y_elastic, c='black', s=100, label="Elastic Net")
plt.legend()
plt.show()


# Iris Dataset with Regularized Logistic Regression
# Load and plot the data
iris = datasets.load_iris()
x = iris['data'][:, (2, 3)]  # Petal length and width
y = iris['target']
plt.scatter(x[:, 0], x[:, 1], cmap='jet', c=y)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

# Create Logistic Regression model with L2 regularization
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
# The lbfgs solver is an optimization algorithm that can handle L2 regularization
log_reg.fit(x, y)
x_log_test = [[2, 0.6], [4.5, 1], [5, 1.7]]

# Make Predictions and Plot
pred_num = 1
for i in x_log_test:
    y_log = log_reg.predict([i])
    y_log_prob = log_reg.predict_proba([i])
    print(f"Class Probabilities for Prediction {pred_num}: {y_log_prob[0]}")
    plt.scatter(i[0], i[1], s=100, label=f"Predicted Class: {y_log[0]}")
    pred_num += 1
plt.legend()
plt.show()
