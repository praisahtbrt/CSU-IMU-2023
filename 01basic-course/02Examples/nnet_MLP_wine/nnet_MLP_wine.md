Can we detect fake wines from their chemical analysis? The data---available from the [UCI](https://archive.ics.uci.edu/) data repository---contain the chemical analysis of three cultivars of Italian wine. We want to develop a model that can automatically class a wine sample into one of the three varieties from the values of its anlaysis.

For this, the datatset has $13$ attributes---chemical compounds---measured on $178$ samples. We will attempt to fit an MLP model for this classification problem.

We begin by loading the data and examining the first few lines.


```python
import pandas as pd
wine = pd.read_csv('wine_data.csv', names = ["Cultivar", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"])
wine.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar</th>
      <th>Alchol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Alcalinity_of_Ash</th>
      <th>Magnesium</th>
      <th>Total_phenols</th>
      <th>Falvanoids</th>
      <th>Nonflavanoid_phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>



As an initial step of the exploratory datta analysis (EDA), we compute the elementary statistics.


```python
wine.describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar</th>
      <td>178.0</td>
      <td>1.938202</td>
      <td>0.775035</td>
      <td>1.00</td>
      <td>1.0000</td>
      <td>2.000</td>
      <td>3.0000</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>Alchol</th>
      <td>178.0</td>
      <td>13.000618</td>
      <td>0.811827</td>
      <td>11.03</td>
      <td>12.3625</td>
      <td>13.050</td>
      <td>13.6775</td>
      <td>14.83</td>
    </tr>
    <tr>
      <th>Malic_Acid</th>
      <td>178.0</td>
      <td>2.336348</td>
      <td>1.117146</td>
      <td>0.74</td>
      <td>1.6025</td>
      <td>1.865</td>
      <td>3.0825</td>
      <td>5.80</td>
    </tr>
    <tr>
      <th>Ash</th>
      <td>178.0</td>
      <td>2.366517</td>
      <td>0.274344</td>
      <td>1.36</td>
      <td>2.2100</td>
      <td>2.360</td>
      <td>2.5575</td>
      <td>3.23</td>
    </tr>
    <tr>
      <th>Alcalinity_of_Ash</th>
      <td>178.0</td>
      <td>19.494944</td>
      <td>3.339564</td>
      <td>10.60</td>
      <td>17.2000</td>
      <td>19.500</td>
      <td>21.5000</td>
      <td>30.00</td>
    </tr>
    <tr>
      <th>Magnesium</th>
      <td>178.0</td>
      <td>99.741573</td>
      <td>14.282484</td>
      <td>70.00</td>
      <td>88.0000</td>
      <td>98.000</td>
      <td>107.0000</td>
      <td>162.00</td>
    </tr>
    <tr>
      <th>Total_phenols</th>
      <td>178.0</td>
      <td>2.295112</td>
      <td>0.625851</td>
      <td>0.98</td>
      <td>1.7425</td>
      <td>2.355</td>
      <td>2.8000</td>
      <td>3.88</td>
    </tr>
    <tr>
      <th>Falvanoids</th>
      <td>178.0</td>
      <td>2.029270</td>
      <td>0.998859</td>
      <td>0.34</td>
      <td>1.2050</td>
      <td>2.135</td>
      <td>2.8750</td>
      <td>5.08</td>
    </tr>
    <tr>
      <th>Nonflavanoid_phenols</th>
      <td>178.0</td>
      <td>0.361854</td>
      <td>0.124453</td>
      <td>0.13</td>
      <td>0.2700</td>
      <td>0.340</td>
      <td>0.4375</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>Proanthocyanins</th>
      <td>178.0</td>
      <td>1.590899</td>
      <td>0.572359</td>
      <td>0.41</td>
      <td>1.2500</td>
      <td>1.555</td>
      <td>1.9500</td>
      <td>3.58</td>
    </tr>
    <tr>
      <th>Color_intensity</th>
      <td>178.0</td>
      <td>5.058090</td>
      <td>2.318286</td>
      <td>1.28</td>
      <td>3.2200</td>
      <td>4.690</td>
      <td>6.2000</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th>Hue</th>
      <td>178.0</td>
      <td>0.957449</td>
      <td>0.228572</td>
      <td>0.48</td>
      <td>0.7825</td>
      <td>0.965</td>
      <td>1.1200</td>
      <td>1.71</td>
    </tr>
    <tr>
      <th>OD280</th>
      <td>178.0</td>
      <td>2.611685</td>
      <td>0.709990</td>
      <td>1.27</td>
      <td>1.9375</td>
      <td>2.780</td>
      <td>3.1700</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>Proline</th>
      <td>178.0</td>
      <td>746.893258</td>
      <td>314.907474</td>
      <td>278.00</td>
      <td>500.5000</td>
      <td>673.500</td>
      <td>985.0000</td>
      <td>1680.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
wine.shape
```




    (178, 14)



### Data Preparation

We should perform a complete EDA, but since we have already decided to fit an MLP, we will skip this stage. The data preparation entails the following steps:

- First, place the data into a data matrix of explanatory variables plus the repsonse variable.
- Then, divide the data into a training set and a test set.
- Finally, normalize the data since they have varying magnitudes. For this, we use the class `StandardScaler` on the training data, which can then be applied to the test data in a `pipeline`. An alternative would be to use the function `scale`  directly.


```python
X = wine.drop('Cultivar',axis=1)
y = wine['Cultivar']
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alchol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Alcalinity_of_Ash</th>
      <th>Magnesium</th>
      <th>Total_phenols</th>
      <th>Falvanoids</th>
      <th>Nonflavanoid_phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.head()
```




    0    1
    1    1
    2    1
    3    1
    4    1
    Name: Cultivar, dtype: int64




```python
# Split into a trainig set and a test set (by defaltt 0.75 / 0.25)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Perform the normalization on the training data
from sklearn.preprocessing import StandardScaler
normaliser = StandardScaler()
normaliser.fit(X_train)
X_train = normaliser.transform(X_train)
X_test  = normaliser.transform(X_test)
X_train[:4,1:6]
```




    array([[ 1.77523647e-01, -2.77539145e-03,  9.10454986e-02,
             1.43559675e+00, -1.02994771e+00],
           [ 1.26873497e+00,  3.41373149e-02,  3.79356244e-01,
            -8.51042387e-01,  2.42670606e-02],
           [-7.81686310e-01, -9.99418462e-01, -1.20635286e+00,
            -1.36467656e-01,  1.86453948e-01],
           [-1.27449143e+00, -3.73095873e+00, -2.61907551e+00,
            -8.51042387e-01, -4.94730979e-01]])




```python
X_test[:4,1:6]
```




    array([[-0.97650873, -1.71293591, -0.26379385, -0.52825887,  0.12081638],
           [ 3.10018413, -0.99556948,  0.47952357, -0.93952772,  0.54887276],
           [-0.70533182,  0.325895  , -1.00711128,  0.56845808,  1.66840483],
           [ 0.72286657,  1.23204207,  1.07417751, -0.18553482, -1.21274389]])



### Train the MLP model

We use the Multi-Layer Perceptron classifier, `MLPClassifier`, from the library `neural_network`


```python
from sklearn.neural_network import MLPClassifier
```

We can now create an instance of the model. 

Among the numerous possible parameters, we only define 

- the number of hidden layers,
- the number od neurons in each hidden layer.

For this, we send a list whose $n$-th element is equal to the number of neurons in hidden layer $n.$ Here we choose $3$ layers, with $13$ neurons each, and we limit the number of iterations to $500.$


```python
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
```

Having defined the model, we can now fit the training data, already prepared and normalized above.


```python
mlp.fit(X_train,y_train)
```




    MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)



The output shows all the default values, as well as the architecture that we defined. All of these could be modified and tuned.


## Predictions and Model Evaluation

With the fitted model, we can now use the method `predict()` to make the actual predictions on the test data and print out the confusion matrix.


```python
previsions = mlp.predict(X_test)
```


```python
# confusion table
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,previsions))
```

    [[16  0  0]
     [ 1 15  0]
     [ 0  0 13]]


We observe $2$ bad lassifications out of $45.$


```python
print(classification_report(y_test,previsions))
```

                  precision    recall  f1-score   support
    
               1       0.94      1.00      0.97        16
               2       1.00      0.94      0.97        16
               3       1.00      1.00      1.00        13
    
        accuracy                           0.98        45
       macro avg       0.98      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45
    


We have an excellent classification rate of $96\%.$

### Conclusions

1. An MLP model with $3$ hidden layers, havinf $13$ neurons each, provides a classifier with an accuracy rate of $96\%.$
2. For a more reliable estimate, we should perform cross-validation.
3. Other supervised learning methods could be used here:
  - k-nn
  - SVM
  - Bagging, etc.
