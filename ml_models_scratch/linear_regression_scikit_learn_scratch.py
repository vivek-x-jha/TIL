#!/usr/bin/env python
# coding: utf-8

# # M01 - Linear Regression

# #### I. Summary
# This is a Notebook showing how to build OOP Linear Rgeression Model inspired by scikit-learn

# #### II. Setup

# In[17]:


import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

np.random.seed = 0


# #### III. Design ABC
# Since I'm writing ML models both with:
# 1. an eye to build more generalized models
# 2. influence from scikit-learn
# 
# I will opt to use a consolidated metaclass `BaseEstimator`

# In[18]:


class BaseEstimator(ABC):
    @abstractmethod
    def __init__(self, normalize=False):
        self._normalize = normalize
        
    @abstractmethod
    def fit(self, design_matrix, labels):
        pass
    
    @abstractmethod
    def predict(self, design_matrix):
        pass
    
    def score(self, design_matrix, labels):
        pass
    
    @abstractmethod
    def get_params(self):
        pass
    
    @abstractmethod
    def set_params(self, params):
        pass


# #### IV. Create support Metrics class

# In[58]:


class Metrics:
    
    @staticmethod
    def normalize(design_matrix, labels):
        """Performs min-max normalization on all columns of data
        y -> (y -  min(y)) / (max(y) - min(y))
        ---------
        Arguments:
        @design_matrix: 1D or 2D numpy array 
        @labels: 1D numpy array        
        ______
        Return: 2-tuple (X, y)
        """        
        print(f'Running normalize')
        return design_matrix, labels
    
    def _residuals(self, design_matrix, labels):
        """Transforms data & labels into residuals
        (X, y) -> y - h(θ; X, y)        
        ---------
        Arguments:
        @design_matrix: 1D or 2D numpy array 
        @labels: 1D numpy array        
        ______
        Return: 1D numpy array (same shape as labels)
        """
        try:
            return labels - self.predict(design_matrix)
        except AttributeError:
            return np.zeros(labels.shape)
        
    def _mse(self, design_matrix, labels):
        """Computes Mean Squared Error of data & labels
        (X, y) -> ∑ (y - h(θ; X, y))^2 / n
        ---------
        Arguments:
        @design_matrix: 1D or 2D numpy array 
        @labels: 1D numpy array        
        ______
        Return: Non-negative Float
        """
        residuals_ = self._residuals(design_matrix, labels)
        return np.mean(residuals_ ** 2)
    
    def _mxe(self, design_matrix, labels):
        """Computes Mean Cross Entropy of data & labels
        (X, y) -> -∑ (y * log(h(θ; X, y)) + (1 - y) * log(1 - h(θ; X, y))) / n
        ---------
        Arguments:
        @design_matrix: 1D or 2D numpy array 
        @labels: 1D numpy array        
        ______
        Return: Non-negative Float
        """
        predictions = self.predict(design_matrix)
        cross_entropy = lambda y, y_: np.log(y_) if y == 1 else np.log(1 - y_)
        cross_entropies = [cross_entropy(y, y_) for y, y_ in zip(labels, predictions)]
        return -np.mean(cross_entropies)

    def _gradient(self, design_matrix, labels):
        """Computes Gradient of cost function
        (X, y) -> ∇J(θ; X, y)
        ---------
        Arguments:
        @design_matrix: 1D or 2D numpy array 
        @labels: 1D numpy array        
        ______
        Return: 1D numpy array (same shape as model coefficients)
        """
        pass
    
    def _gradient_descent(self, design_matrix, labels, cost_func, cost_threshold=1e-5, learn_rate=.05):
        """Performs gradient descent
        1. Initialize random array of coefficients
        2. Initialize delta_cost
        3. Loop and perform gradient descent
        ---------
        Arguments:
        @design_matrix: 1D or 2D numpy array
        @labels: 1D numpy array
        @cost_func: function that takes 2 positional arguments
        
        Keyword Arguments:
        @cost_threshold (default=1e-5)
        @learn_rate (default=.05)
        ______
        Return: None
        """
        rand_array = np.random.rand(design_matrix.shape[1] + self._fit_intercept)
        self.set_params(rand_array)
        
        delta_cost = 1
        cost = cost_func(design_matrix, labels)
        self._cost_sequence = [cost]
        while delta_cost > cost_threshold:
            old_cost = cost
            new_params = self.get_params() - learn_rate * self._gradient(design_matrix, labels)
            self.set_params(new_params)
            cost = cost_func(design_matrix, labels)
            self._cost_sequence.append(cost)
            delta_cost = cost - old_cost
        


# #### V. Defining the Linear Regression Model class

# In[59]:


class LinearRegression(BaseEstimator, Metrics):
    def __init__(self, fit_intercept=True, normalize=False):
        super().__init__(normalize=normalize)
        self._fit_intercept = fit_intercept
        
    def _prepend_ones(self, design_matrix):
        ones_array = np.ones(design_matrix.shape[0])
        if self._fit_intercept:
            return np.insert(design_matrix, 0, ones_array, axis=1)
        return design_matrix
    
    def _gradient(self, design_matrix, labels):
        X = self._prepend_ones(design_matrix)
        residuals_ = self._residuals(design_matrix, labels)
        return (-2 / X.shape[0]) * np.dot(X.T, residuals_)
    
    def fit(self, design_matrix, labels):
        X, y = type(self).normalize(design_matrix, labels)
        self._gradient_descent(X, y, self._mse)
        return self
    
    def predict(self, design_matrix):
        X = self._prepend_ones(design_matrix)
        return np.dot(X, self.get_params())
    
    def score(self, design_matrix, labels):
        r_squared_ = 1 - self._mse(design_matrix, labels) / np.var(labels)
        return r_squared_
    
    def get_params(self):
        return self.coefs_
    
    def set_params(self, params):
        self.coefs_ = params
    
    


# #### VI. Testing the class and methods

# In[93]:


# Setup sample/random data
sample_size = 1000
num_features = 6
X = np.around(np.random.rand(sample_size * num_features).reshape(sample_size, num_features) * 100, 2)
y = np.around(np.random.rand(sample_size) * 100, 2)

X = np.array([[1, 3], [-3, 3], [4, 5], [2, 3], [-1, 7], [8, 9]])
y = np.array([1, 2, 4, 2, 5, 6])


# In[94]:


reg = LinearRegression().fit(X, y)
reg.get_params()


# #### VII. Extending to Logistic Regression

# In[95]:


class LogisticRegression(LinearRegression):
    def __init__(self, fit_intercept=True, normalize=False):
        super().__init__(fit_intercept=fit_intercept, normalize=normalize)
        
    def _gradient(self, design_matrix, labels):
        return super()._gradient(design_matrix, labels) / 2
        
    def fit(self, design_matrix, labels):
        X, y = type(self).normalize(design_matrix, labels)
        self._gradient_descent(X, y, self._mxe)
        return self
    
    def predict(self, design_matrix):
        predictions = super().predict(design_matrix)
        sigmoid = lambda t: 1 / (1 + np.exp(-t))
        return sigmoid(predictions)
    
    def score(self, design_matrix, labels):
        return self._mxe(design_matrix, labels)
        
    def classify(self, design_matrix, decision_boundary=0.5):
        predictions = self.predict(design_matrix)
        classifications = [int(prediction >= decision_boundary) for prediction in predictions]
        return np.array(classifications)
        


# In[132]:


y = np.array([1, 0, 1, 1, 0, 0])
clf = LogisticRegression().fit(X, y)
print(clf._cost_sequence)


# In[ ]:




