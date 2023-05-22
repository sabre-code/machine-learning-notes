---
jupyter:
  kernelspec:
    display_name: ml_book
    language: python
    name: python3
  language_info:
    name: python
    version: 3.11.3
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
---

::: {.cell .markdown}
1.  Implementation of Perceptron in Python
:::

::: {.cell .code}
``` python
import numpy as np

class Perceptron:
    """Perceptron classifier
    
    Parameters
    ----------
    eta : float
        Learning Rate (betweeen 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight
        initialization

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting

    errors_ : list
        Number of misclassifications (updates) in each epoch.


    """

    def _init_(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of 
            examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.

        Returns
        -------
        self : object

        We will attempt to train the perceptron using these two points and update the weights based on the perceptron learning rule:

        Initialize the weights and bias to zero: w₁ = 0, w₂ = 0, b = 0.

        For the first training point, calculate the output:
        z = (w₁ * x₁) + (w₂ * x₂) + b
        = (0 * 1) + (0 * 1) + 0
        = 0
        Apply the step function to the weighted sum:
        output = step(0)
        = 0
        The predicted output is 0, which does not match the target output of 1.

        Update the weights using the perceptron learning rule:
        Δw₁ = η * (target output - predicted output) * x₁
        = η * (1 - 0) * 1
        = η
        Δw₂ = η * (target output - predicted output) * x₂
        = η * (1 - 0) * 1
        = η
        Δb = η * (target output - predicted output)
        = η * (1 - 0)
        = η
        (Here, η is the learning rate.)

        Since the predicted output is 0 and the target output is 1, the weight updates become:
        w₁ = w₁ + Δw₁ = 0 + η
        w₂ = w₂ + Δw₂ = 0 + η
        b = b + Δb = 0 + η
        
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))  #Perceptron Learning Rule Δw and Δb
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X,self.w_) + self.b_
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0) 
```
:::

::: {.cell .markdown}
The perceptron is a binary classifier. However the perceptron algorithm
can be extended to multiclass classification using OvA (One versus all)
or OvR (One versus rest).Here we train n classifiers where n is number
of classes. Final class label is assignedto new examples based on the
highest probability output among the n classifiers.Each classifier is
trained positive for on class and negative to all other.
:::

::: {.cell .markdown}
:::
