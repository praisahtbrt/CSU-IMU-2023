# k-NN Classifier for IRIS Data

We will fit a *k*-nn classifier to Fisher’s `iris` data. The Iris
dataset contains measurements of sepal and petal lengths and widths for
3 different species. Each row in the dataset contains the 4 measurements
and the species of the measured specimen. There is a total of 150
specimens/individuals.

We begin by loading the data and performing an exhaustive exploratory
data analysis (EDA), where we calculate all the elementary statistics,
plot the data in various forms and then prepare the data for the
supervised learning analysis.

``` r
# Read in `iris` data
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE) 
# Print first lines
head(iris)
```

    ##    V1  V2  V3  V4          V5
    ## 1 5.1 3.5 1.4 0.2 Iris-setosa
    ## 2 4.9 3.0 1.4 0.2 Iris-setosa
    ## 3 4.7 3.2 1.3 0.2 Iris-setosa
    ## 4 4.6 3.1 1.5 0.2 Iris-setosa
    ## 5 5.0 3.6 1.4 0.2 Iris-setosa
    ## 6 5.4 3.9 1.7 0.4 Iris-setosa

``` r
# Add column names
names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")
# Check the result
head(iris)
```

    ##   Sepal.Length Sepal.Width Petal.Length Petal.Width     Species
    ## 1          5.1         3.5          1.4         0.2 Iris-setosa
    ## 2          4.9         3.0          1.4         0.2 Iris-setosa
    ## 3          4.7         3.2          1.3         0.2 Iris-setosa
    ## 4          4.6         3.1          1.5         0.2 Iris-setosa
    ## 5          5.0         3.6          1.4         0.2 Iris-setosa
    ## 6          5.4         3.9          1.7         0.4 Iris-setosa

Scatter plots, by species

``` r
library(ggplot2)
ggplot(iris, aes(x=Sepal.Length, y=Sepal.Width, shape =Species, color=Species)) +
  geom_point()
```

![](k_nn_iris_md_files/figure-markdown_github/unnamed-chunk-2-1.png)
There is a good correlation for Setosa.

Look at petals.

``` r
ggplot(iris, aes(x=Petal.Length, y=Petal.Width, shape =Species, color=Species)) +
  geom_point()
```

![](k_nn_iris_md_files/figure-markdown_github/unnamed-chunk-3-1.png)

All three species have positive correlations. We look at these more
closely.

``` r
# Overall correlation `Petal.Length` and `Petal.Width`
cor(iris$Petal.Length, iris$Petal.Width)
```

    ## [1] 0.9627571

``` r
# Return values of `iris` levels 
x=levels(as.factor(iris$Species))
# Print Setosa correlation matrix
print(x[1])
```

    ## [1] "Iris-setosa"

``` r
cor(iris[iris$Species==x[1],1:4])
```

    ##              Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## Sepal.Length    1.0000000   0.7467804    0.2638741   0.2790916
    ## Sepal.Width     0.7467804   1.0000000    0.1766946   0.2799729
    ## Petal.Length    0.2638741   0.1766946    1.0000000   0.3063082
    ## Petal.Width     0.2790916   0.2799729    0.3063082   1.0000000

``` r
# Print Versicolor correlation matrix
print(x[2])
```

    ## [1] "Iris-versicolor"

``` r
cor(iris[iris$Species==x[2],1:4])
```

    ##              Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## Sepal.Length    1.0000000   0.5259107    0.7540490   0.5464611
    ## Sepal.Width     0.5259107   1.0000000    0.5605221   0.6639987
    ## Petal.Length    0.7540490   0.5605221    1.0000000   0.7866681
    ## Petal.Width     0.5464611   0.6639987    0.7866681   1.0000000

``` r
# Print Virginica correlation matrix
print(x[3])
```

    ## [1] "Iris-virginica"

``` r
cor(iris[iris$Species==x[3],1:4])
```

    ##              Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## Sepal.Length    1.0000000   0.4572278    0.8642247   0.2811077
    ## Sepal.Width     0.4572278   1.0000000    0.4010446   0.5377280
    ## Petal.Length    0.8642247   0.4010446    1.0000000   0.3221082
    ## Petal.Width     0.2811077   0.5377280    0.3221082   1.0000000

Check the repartitioning of the samples among the species.

``` r
# Division of `Species`
table(iris$Species) 
```

    ## 
    ##     Iris-setosa Iris-versicolor  Iris-virginica 
    ##              50              50              50

``` r
# Percentage division of `Species`
round(prop.table(table(iris$Species)) * 100, digits = 1)
```

    ## 
    ##     Iris-setosa Iris-versicolor  Iris-virginica 
    ##            33.3            33.3            33.3

Summary statistics:

``` r
# Summary overview of `iris`
summary(iris) 
```

    ##   Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
    ##  Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
    ##  1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
    ##  Median :5.800   Median :3.000   Median :4.350   Median :1.300  
    ##  Mean   :5.843   Mean   :3.054   Mean   :3.759   Mean   :1.199  
    ##  3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
    ##  Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  
    ##    Species         
    ##  Length:150        
    ##  Class :character  
    ##  Mode  :character  
    ##                    
    ##                    
    ## 

``` r
# Refined summary overview
summary(iris[c("Petal.Width", "Sepal.Width")])
```

    ##   Petal.Width     Sepal.Width   
    ##  Min.   :0.100   Min.   :2.000  
    ##  1st Qu.:0.300   1st Qu.:2.800  
    ##  Median :1.300   Median :3.000  
    ##  Mean   :1.199   Mean   :3.054  
    ##  3rd Qu.:1.800   3rd Qu.:3.300  
    ##  Max.   :2.500   Max.   :4.400

To use *k*-nn, we must load the `class` library

``` r
library(class)
```

Though not strictly necessary here, we should normmalize the data before
applying *k*-nn. 

``` r
# Build your own `normalize()` function
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}
# Normalize the `iris` data
iris_norm <- as.data.frame(lapply(iris[1:4], normalize))
# Summarize `iris_norm`
summary(iris_norm)
```

    ##   Sepal.Length     Sepal.Width      Petal.Length     Petal.Width     
    ##  Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   Min.   :0.00000  
    ##  1st Qu.:0.2222   1st Qu.:0.3333   1st Qu.:0.1017   1st Qu.:0.08333  
    ##  Median :0.4167   Median :0.4167   Median :0.5678   Median :0.50000  
    ##  Mean   :0.4287   Mean   :0.4392   Mean   :0.4676   Mean   :0.45778  
    ##  3rd Qu.:0.5833   3rd Qu.:0.5417   3rd Qu.:0.6949   3rd Qu.:0.70833  
    ##  Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   Max.   :1.00000

The next important step is to divide the data into training and test
sets. We use the function `sample` after an initialization of the random
number seed.

``` r
set.seed(12345)
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))
# Extract training set
iris.training <- iris[ind==1, 1:4]
# Inspect training set
head(iris.training)
```

    ##    Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## 5           5.0         3.6          1.4         0.2
    ## 6           5.4         3.9          1.7         0.4
    ## 7           4.6         3.4          1.4         0.3
    ## 8           5.0         3.4          1.5         0.2
    ## 11          5.4         3.7          1.5         0.2
    ## 12          4.8         3.4          1.6         0.2

``` r
# Extract test set
iris.test <- iris[ind==2, 1:4]
# Inspect test set
head(iris.test)
```

    ##    Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## 1           5.1         3.5          1.4         0.2
    ## 2           4.9         3.0          1.4         0.2
    ## 3           4.7         3.2          1.3         0.2
    ## 4           4.6         3.1          1.5         0.2
    ## 9           4.4         2.9          1.4         0.2
    ## 10          4.9         3.1          1.5         0.1

Now put the corresponding labels into two vectors.

``` r
# Extract `iris` training labels
iris.trainLabels <- iris[ind==1,5]
# Inspect result
#print(iris.trainLabels)
# Extract `iris` test labels
iris.testLabels <- iris[ind==2, 5]
# Inspect result
#print(iris.testLabels)
```

We are ready to build the classifier.

``` r
# Build the model
iris_pred <- knn(train = iris.training, test = iris.test, cl = iris.trainLabels, k=3)
# Inspect the result `iris_pred`
head(iris_pred)
```

    ## [1] Iris-setosa Iris-setosa Iris-setosa Iris-setosa Iris-setosa Iris-setosa
    ## Levels: Iris-setosa Iris-versicolor Iris-virginica

But we need to evaluate the model for its precision by comparing
observed and predicted classes in the test set.

``` r
# Put `iris.testLabels` in a data frame
irisTestLabels <- data.frame(iris.testLabels)
# Merge `iris_pred` and `iris.testLabels` 
merge <- data.frame(iris_pred, iris.testLabels)
# Specify column names for `merge`
names(merge) <- c("Predicted Species", "Observed Species")
# Inspect `merge` 
head(merge)
```

    ##   Predicted Species Observed Species
    ## 1       Iris-setosa      Iris-setosa
    ## 2       Iris-setosa      Iris-setosa
    ## 3       Iris-setosa      Iris-setosa
    ## 4       Iris-setosa      Iris-setosa
    ## 5       Iris-setosa      Iris-setosa
    ## 6       Iris-setosa      Iris-setosa

It is easier to inspect using a confusion matrix. We will use

-   the simplest version, and a
-   more sophisticated table from the library `gmodels`

``` r
table(iris_pred,iris.testLabels)
```

    ##                  iris.testLabels
    ## iris_pred         Iris-setosa Iris-versicolor Iris-virginica
    ##   Iris-setosa              18               0              0
    ##   Iris-versicolor           0              18              1
    ##   Iris-virginica            0               2             21

``` r
library(gmodels)
CrossTable(x = iris.testLabels, y = iris_pred, prop.chisq=FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  60 
    ## 
    ##  
    ##                 | iris_pred 
    ## iris.testLabels |     Iris-setosa | Iris-versicolor |  Iris-virginica |       Row Total | 
    ## ----------------|-----------------|-----------------|-----------------|-----------------|
    ##     Iris-setosa |              18 |               0 |               0 |              18 | 
    ##                 |           1.000 |           0.000 |           0.000 |           0.300 | 
    ##                 |           1.000 |           0.000 |           0.000 |                 | 
    ##                 |           0.300 |           0.000 |           0.000 |                 | 
    ## ----------------|-----------------|-----------------|-----------------|-----------------|
    ## Iris-versicolor |               0 |              18 |               2 |              20 | 
    ##                 |           0.000 |           0.900 |           0.100 |           0.333 | 
    ##                 |           0.000 |           0.947 |           0.087 |                 | 
    ##                 |           0.000 |           0.300 |           0.033 |                 | 
    ## ----------------|-----------------|-----------------|-----------------|-----------------|
    ##  Iris-virginica |               0 |               1 |              21 |              22 | 
    ##                 |           0.000 |           0.045 |           0.955 |           0.367 | 
    ##                 |           0.000 |           0.053 |           0.913 |                 | 
    ##                 |           0.000 |           0.017 |           0.350 |                 | 
    ## ----------------|-----------------|-----------------|-----------------|-----------------|
    ##    Column Total |              18 |              19 |              23 |              60 | 
    ##                 |           0.300 |           0.317 |           0.383 |                 | 
    ## ----------------|-----------------|-----------------|-----------------|-----------------|
    ## 
    ## 

Alternatively, we can use the comprehensive `caret` library for
classification and regression training.

``` r
library(caret)
```

    ## Loading required package: lattice

``` r
library(e1071)
# Create index to split based on labels  
index <- createDataPartition(iris$Species, p=0.75, list=FALSE)
# Subset training set with index
iris.training <- iris[index,]
# Subset test set with index
iris.test <- iris[-index,]
# Overview of the 237 algorithms supported by caret
#names(getModelInfo())
# Train a model using k-nn
model_knn <- train(iris.training[, 1:4], iris.training[, 5], method='knn')
```

With the trained model, we can now predict the test lablels and check
the accuracy.

``` r
# Predict the labels of the test set
predictions<-predict(object=model_knn,iris.test[,1:4])
# Evaluate the predictions
table(predictions)
```

    ## predictions
    ##     Iris-setosa Iris-versicolor  Iris-virginica 
    ##              12              13              11

``` r
# Confusion matrix 
confusionMatrix(as.factor(predictions),as.factor(iris.test[,5]))
```

    ## Confusion Matrix and Statistics
    ## 
    ##                  Reference
    ## Prediction        Iris-setosa Iris-versicolor Iris-virginica
    ##   Iris-setosa              12               0              0
    ##   Iris-versicolor           0              12              1
    ##   Iris-virginica            0               0             11
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9722          
    ##                  95% CI : (0.8547, 0.9993)
    ##     No Information Rate : 0.3333          
    ##     P-Value [Acc > NIR] : 4.864e-16       
    ##                                           
    ##                   Kappa : 0.9583          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Iris-setosa Class: Iris-versicolor
    ## Sensitivity                      1.0000                 1.0000
    ## Specificity                      1.0000                 0.9583
    ## Pos Pred Value                   1.0000                 0.9231
    ## Neg Pred Value                   1.0000                 1.0000
    ## Prevalence                       0.3333                 0.3333
    ## Detection Rate                   0.3333                 0.3333
    ## Detection Prevalence             0.3333                 0.3611
    ## Balanced Accuracy                1.0000                 0.9792
    ##                      Class: Iris-virginica
    ## Sensitivity                         0.9167
    ## Specificity                         1.0000
    ## Pos Pred Value                      1.0000
    ## Neg Pred Value                      0.9600
    ## Prevalence                          0.3333
    ## Detection Rate                      0.3056
    ## Detection Prevalence                0.3056
    ## Balanced Accuracy                   0.9583
