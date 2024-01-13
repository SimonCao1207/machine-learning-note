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
    - [Logistic regression](#logistic-regression)
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
or 
$$f(x) = w^Tx$$
- Loss function: 
$$L(w) = \frac{1}{2}\sum_{i=1}^N(y_i - \bar{x_i}w)^2 =\frac{1}{2}\ \| y-\bar{X}w \|_2^2 $$
(check out [norm](#norm) in Math)
- $\frac{dL}{dw} = \bar{X}^T(\bar{X}w-y) = 0 \Leftrightarrow \bar{X}^T\bar{X}w = \bar{X}^Ty \triangleq b$

- If $A = \bar{X}^T\bar{X}$ is invertible, $w = A^{-1}b$
> **_Question_** : Why don't we use absolute instead of square in loss function ?
> - Square function has a well-defined derivative everywhere, meanwhile absolute function has a non-differentiable point at 0.

> **_Question_** :  What if $A = \bar{X}^T\bar{X}$ is not invertible ? 
> - [Pseudo inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)

## Logistic regression
- a process of modeling the probability of a discrete outcome given an input variable.  
$$f(x) = \theta(w^Tx)$$
where $\theta$ is [activation function](#activation-function) that outputs a number between [0,1].
- For training set $X = [x_1, x_2, ..., x_N] \in R^{d \times N}$ and $y=[y_1, y_2, y_3, ..., y_N]$, the objective is to find $w$ for $P(y|X;w)$ to maximize

- This is maximum likelihood estimation problem with $P(y|X;w)$ as a likelihood function: 

$$w=\underset{w}{argmax}(P(y|X;w))$$

- Let $z_i=\theta(w^Tx_i)$ and assume from now for simplicity (binary classification) $y_i\in\set{0,1}$ then:
$$P(y_i|x_i;w) = z_i^{y_i}(1-z_i)^{1-y_i}$$

- Loss function (build from likelihood function): 
$$J(w) = -log(P(y|X;w))$$
$$=-\sum_{i=1}^N(y_ilog(z_i)+(1-y_i)log(1-z_i))$$
- Derivative with regards to $w$: 
    $$\frac{dz_i}{dw} = (z_i-y_i)x_i$$

- Update formula following [SGD](#stochastic-gradient-descent):
    $$w = w + \eta(z_i-y_i)x_i $$
- Property: 
    - boundary created by logistic regression is a hyperplane $w^Tx$. Therefore, this model only works for data with 2 classes are almost linearly separable.

## Gradient descent
- Gradient points in the direction of the steepest increase in the loss

$$\theta = \theta - \eta\nabla_{\theta}J(\theta)$$

- **Key idea**: update the parameters in the opposite direction of the gradient (derivative) of the loss function with respect to those parameters. 

> **_Question_** :  What is gradient descent ? 
> - Gradient descent is an optimization algorithm to minimize the loss function by updating the model's parameters 
> - The learning rate is a hyperparameter that determines the size of the steps taken during the parameter updates.

### Batch Gradient Descent
- computes the gradient using the whole dataset

### Stochastic Gradient Descent
    
- picks a random instance in the training set at every step and computes the gradients based only on that single instance. 
    $$\theta = \theta - \eta\nabla_{\theta}J(\theta;x_i;y_i)$$
    > **_Question_** : What is advantage and disadvantage of SGD ? 
    > - Advantage: 
        - faster since it has little data to manipulate at every iteration $\rightarrow$ make it possible to train on a huge training sets. 
        - Due to its stochastic (i.e., random) nature, is good to escape from local optimai and has a better chance of finding the global minimum than Batch Gradient descent 
    > - Disadvantage: randomness means that the algorithm can never settle at the minimum and continue to bounce around
        - one solution is gradually reduce the learning rate.
### Mini-batch Gradient Descent
- number of picked instances > 1 (but still a lot fewer than $N$)

### Stopping Criteria
> **_Question_** : When do we know the algorithm is converged and should stop ? 
- In practice, there are a few number of ways.
    - predetermined maximum number of iterations. $\rightarrow$ can stop too soon
    - stop when the norm of the gradient is below some threshold
    $$\nabla_{\theta}J(w) < \epsilon$$
    - stop when the improvement drops below a threshold $\rightarrow$ might stuck in "saddle points"

### Gradient descent with momentum 
- It cares about what previous gradients were: 
    $$m \leftarrow \beta m + \eta\nabla_{\theta}J(\theta)$$
    - to simulate some sort of friction mechanism and prevent the momentum from growing too large, the algorithm introduces a new hyperparameter $\beta$, called the momentum, which must be set between 0 (high friction) and 1(no friction). A typical momentum value is 0.9.
    $$\theta \leftarrow \theta - m$$

# Math
Some math knowledge needed about Linear algebra, Probability, Optimization, Discrete math, ... necessary for understanding of machine learning.

## Norm

## Activation function
- Sigmoid: 
    $$\sigma(x) = \frac{1}{e^{-x} +1}$$
    - specially, $\sigma^{'}(x) = \sigma(x)(1-\sigma(x))$
- Tanh:
    $$\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$$
    - specially, $\tanh(x) = 2\sigma(2x)-1$


## Reference 
1. [Machine learning co ban](https://machinelearningcoban.com/2016/12/27/categories/)
2. [A tour of ML algorithms](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)
3. [Aurélien Géron - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)










