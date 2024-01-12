# Machine learning note

## Group based on learning style

- **Supervised learning**: pair of (data, label) is known
    - Classification
    - Regression
- **Unsupervised learning**: 
    - Clustering
    - Association
- **Semi-supervised learning**: only a portion of the data are labeled. 
- **Reinforcement learning** : 

## Algorithms group by similarity
- **Regression algorithms:**
    - [Linear regression](#linear-regression)
    - Logistic regression
    - Stepwise regression

- **Classification algorithm:** 
    - Linear classifier
    - Support vector machine (SVM)
    - Kernel SVM
    - Sparse representation-based classification (SRC)

- **Instance-based algorithm:**
    - k-nearest neighbors (kNN):
    - Learning Vector Quantization (LVQ)

- **Regularization algorithm:**
    - Ridge regression
    - Least absolute shrinkage and selection operator (LASSO)
    - Least-Angle Regression (LARS)
- **Bayesian algorithm:**
    - Naive Bayes
    - Gaussian Naive Bayes
- **Clustering algorithm:** 
    - k-means clustering
    - k-medians 
    - Expectation maximization (EM)
- **Dimensionality reduction algorithms:** 
    - Principle component analysis (PCA)
    - Linear discriminant analysis (LDA)
- **Ensemble algorithm:**
    - Boosting 
    - Adaboost
    - Random forest
- **Deep learning algorithms:**
    - Perceptron
    - Softmax regression
    - Multi-layer Perceptron    
    - Back-propagation

## Linear regression
$y$ (real value of outcome) and $\hat{y}$ (prediction value of outcome) are scalars.

$\bar{x} = [1, x_1, x_2, ..., x_N]$ is a row vector contains input information.

$w=[w_0, w_1, w_2, ..., w_N]^T$ is a column vector that need to be optimized, $w_0$ is called bias.

$$y \approx \bar{x}w = \hat{y}$$
- Loss function: 
$$L(w) = \frac{1}{2}\sum_{i=1}^N(y_i - \bar{x_i}w)^2 =\frac{1}{2}\ \| y-\bar{X}w \|_2^2 $$
(check out [norm](#norm) in Math)
- $\frac{dL}{dw} = \bar{X}^T(\bar{X}w-y) = 0 \Leftrightarrow \bar{X}^T\bar{X}w = \bar{X}^Ty \triangleq b$

- If $A = \bar{X}^T\bar{X}$ is invertible, $w = A^{-1}b$

- **Question:** Why don't we use absolute instead of square in loss function ?
    - Square function has a well-defined derivative everywhere, meanwhile absolute function has a non-differentiable point at 0.

- **Question:** What if $A = \bar{X}^T\bar{X}$ is not invertible ? 
    - [Pseudo inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)


# Math
Some math knowledge needed about Linear algebra, Probability, Optimization, Discrete math, ... necessary for understanding of machine learning.

## Norm

## Reference 
1. [Machine learning co ban](https://machinelearningcoban.com/2016/12/27/categories/)
2. [A tour of ML algorithms](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)










