# k-NN and cross-validation
- Validation
- k-Fold Cross Validation
- Leave-one-out cross-validation

#### Load Iris dataset
The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant
- Iris Setosa
- Iris Versicolour
- Iris Virginica


```python
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

print(X.shape) # sepal length, width, petal length and width in cm
print(y.shape)
```

    (150, 4)
    (150,)


### Set up a k-NN model

Define a model for classification with NN = 1.


```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
```

### Validation
Hold out set is important to test the model's performance


```python
from sklearn.model_selection import train_test_split # split the data into 60% training and 40% validation

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, train_size=0.6)

# fit the model on the training data
model.fit(X_train, y_train)

# evaluate the model on the validation data
y_val_predicted = model.predict(X_val)
y_val_predicted
```




    array([2, 1, 0, 2, 0, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1,
           0, 0, 2, 0, 0, 1, 1, 0, 2, 1, 0, 2, 2, 1, 0, 2, 1, 1, 2, 0, 2, 0,
           0, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2])



### Precision

Let's see how good we did? Calculate the overall precision.


```python
from sklearn.metrics import accuracy_score
accuracy_score(y_val, y_val_predicted)# The hold-out set is similar to unknown data, because the model has not "seen" it before.
```




    0.9166666666666666



### k-fold CV

We try 5-fold CV for better stability.


```python
from sklearn.model_selection import cross_val_score
K = 5
cross_val_score(model, X, y, cv=K)
```




    array([0.96666667, 0.96666667, 0.93333333, 0.93333333, 1.        ])



### Model validation using leave One Out Validation
This is a special case of k-fold cross-validation where the number of folds equals the number of instances in the data set.


```python
import numpy as np
L = len(np.unique(y))
conf_matrix = np.zeros((L,L))
conf_matrix
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(X)
```




    150




```python
for train_index, val_index in loo.split(X):
#    print(test_index)
   X_train, X_val = X[train_index], X[val_index]
   y_train, y_val = y[train_index], y[val_index]
   model.fit(X_train, y_train)
   y_val_predicted = model.predict(X_val)
   conf_matrix[y_val,y_val_predicted] +=1
```


```python
conf_matrix
```




    array([[50.,  0.,  0.],
           [ 0., 47.,  3.],
           [ 0.,  3., 47.]])




```python
TP = sum(conf_matrix[i][i] for i in range(L))
TP
```




    144.0




```python
accuracy = TP/len(y)
accuracy
```




    0.96




```python

```
