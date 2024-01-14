# Deep learning for computer vision

## Generative Adversarial Networks (Simple GAN)
[paper](https://arxiv.org/abs/1406.2661)
- simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G
- The training procedure for G is to maximize the probability of D making a mistake. 
- The models are both multilayer perceptrons.
-  **_NOTATION_** : 
    - $G(z, \theta_g)$ : multilayer perceptron with parameters $\theta_g$, input noise $z$

    - $p_g$ : generator's distribution over data $x$


    - $D(x, \theta_g)$ : multilayer perceptron with parameters $\theta_g$ (output a single scalar). Represent the probability that $x$ came from that data rather than $p_g$

- Train $D$ to maximize probability of assigning the correct label to both training examples and samples from $G$.
    $$\frac{1}{m} \sum_{i=1}^{m}[logD(x^{i}) + log(1-D(G(z^{(i)})))]$$
- Train $G$ to minimize $log(1-D(G(z)))$
    $$\frac{1}{m} \sum_{i=1}^{m}log(1-D(G(z^{(i)})))$$
    - In practice, generator $G$ is trained to instead
        $$\underset{G}{max}\mathbb(E)_{z\sim p_z}(D(G(z)))$$
    - This new loss function leads to non-saturating gradients


## Diffusion model