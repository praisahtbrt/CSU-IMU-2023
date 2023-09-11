We show the use of 5 resampling methods on the `iris` data, with the
most simple classifier model, *naive Bayes.* The methods used are:

1.  Train-test split.
2.  Bootstrap.
3.  k-fold CV.
4.  k-fold CV with repeats.
5.  LOOCV.

## Data

Load the data and split into train-test with a ratio 80-20.

``` r
# load the libraries
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(klaR) # required for 'naive Bqyes'
```

    ## Loading required package: MASS

``` r
# load  `iris` data
data(iris)
# define the train/test split as 80%/20% 
split=0.80
trainIndex <- createDataPartition(iris$Species, p=split, list=FALSE)
data_train <- iris[ trainIndex,]
data_test  <- iris[-trainIndex,]
# fit a "naive bayes" model
model <- NaiveBayes(Species~., data=data_train)
# predictions on the test set
x_test <- data_test[,1:4]
y_test <- data_test[,5]
predictions <- predict(model, x_test)
# print the results
confusionMatrix(predictions$class, y_test)
```

    ## Confusion Matrix and Statistics
    ## 
    ##             Reference
    ## Prediction   setosa versicolor virginica
    ##   setosa         10          0         0
    ##   versicolor      0         10         0
    ##   virginica       0          0        10
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.8843, 1)
    ##     No Information Rate : 0.3333     
    ##     P-Value [Acc > NIR] : 4.857e-15  
    ##                                      
    ##                   Kappa : 1          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: setosa Class: versicolor Class: virginica
    ## Sensitivity                 1.0000            1.0000           1.0000
    ## Specificity                 1.0000            1.0000           1.0000
    ## Pos Pred Value              1.0000            1.0000           1.0000
    ## Neg Pred Value              1.0000            1.0000           1.0000
    ## Prevalence                  0.3333            0.3333           0.3333
    ## Detection Rate              0.3333            0.3333           0.3333
    ## Detection Prevalence        0.3333            0.3333           0.3333
    ## Balanced Accuracy           1.0000            1.0000           1.0000

## Resampling by Bootstrap

Sampling with replacement.

``` r
# define control parameters for the training
train_control <- trainControl(method="boot", number=100)
# fit the model
model <- train(Species~., 
               data=iris, 
               trControl=train_control, 
               method="nb")
# print the results
print(model)
```

    ## Naive Bayes 
    ## 
    ## 150 samples
    ##   4 predictor
    ##   3 classes: 'setosa', 'versicolor', 'virginica' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (100 reps) 
    ## Summary of sample sizes: 150, 150, 150, 150, 150, 150, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   usekernel  Accuracy   Kappa    
    ##   FALSE      0.9524187  0.9278325
    ##    TRUE      0.9553151  0.9322328
    ## 
    ## Tuning parameter 'fL' was held constant at a value of 0
    ## Tuning
    ##  parameter 'adjust' was held constant at a value of 1
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were fL = 0, usekernel = TRUE and adjust
    ##  = 1.

## k-fold Cross-Validation

We use the deafult, 10-fold CV.

``` r
# define control parameters for the training
train_control <- trainControl(method="cv", number=10)
# fix tuning parameters of the algorithm
grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE),.adjust=FALSE)
# fit the model
model <- train(Species~., 
               data=iris, 
               trControl=train_control, 
               method="nb", 
               tuneGrid=grid)
# print the results
print(model)
```

    ## Naive Bayes 
    ## 
    ## 150 samples
    ##   4 predictor
    ##   3 classes: 'setosa', 'versicolor', 'virginica' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 135, 135, 135, 135, 135, 135, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa
    ##   0.9533333  0.93 
    ## 
    ## Tuning parameter 'fL' was held constant at a value of 0
    ## Tuning
    ##  parameter 'usekernel' was held constant at a value of FALSE
    ## Tuning
    ##  parameter 'adjust' was held constant at a value of FALSE

## Repeated k-fold Cross-Validation

10-fold with 3 repeats.

``` r
# define control parameters for the training
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)
# fit the model
model <- train(Species~., 
               data=iris, 
               trControl=train_control, 
               method="nb")
# print the results
print(model)
```

    ## Naive Bayes 
    ## 
    ## 150 samples
    ##   4 predictor
    ##   3 classes: 'setosa', 'versicolor', 'virginica' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 135, 135, 135, 135, 135, 135, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   usekernel  Accuracy   Kappa    
    ##   FALSE      0.9533333  0.9300000
    ##    TRUE      0.9577778  0.9366667
    ## 
    ## Tuning parameter 'fL' was held constant at a value of 0
    ## Tuning
    ##  parameter 'adjust' was held constant at a value of 1
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were fL = 0, usekernel = TRUE and adjust
    ##  = 1.

## LOOCV

``` r
# define control parameters for the training
train_control <- trainControl(method="LOOCV")
# fit the model
model <- train(Species~., 
               data=iris, 
               trControl=train_control, 
               method="nb")
# print the results
print(model)
```

    ## Naive Bayes 
    ## 
    ## 150 samples
    ##   4 predictor
    ##   3 classes: 'setosa', 'versicolor', 'virginica' 
    ## 
    ## No pre-processing
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 149, 149, 149, 149, 149, 149, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   usekernel  Accuracy   Kappa
    ##   FALSE      0.9533333  0.93 
    ##    TRUE      0.9600000  0.94 
    ## 
    ## Tuning parameter 'fL' was held constant at a value of 0
    ## Tuning
    ##  parameter 'adjust' was held constant at a value of 1
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were fL = 0, usekernel = TRUE and adjust
    ##  = 1.
