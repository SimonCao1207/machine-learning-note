# Deep learning for computer vision

## Convolution Neural network (CNN)

<img src="../img/cnn.png" width="400">

### Pooling layer
- Common types: Max Pooling, Average Pooling
> **_Question_** : Why pooling layer is needed ? 
> - Reduce the size (spatial dimensions) of feature map
> - Increase receptive field : allowing them to capture more global features. 
> - Creating a level of invariance to small translations
### Receptive Field
- The input size of a kernel (prev layer or original input)
- Small RF misses important info, big RF cannot capture locality (overfit).
### Stride
- How many pixels/features to move to get the next receptive field $\rightarrow$ dimension reduction.
### Padding
- Zero padding
- Valid padding
- Same padding 
- Formula:  $$\frac{W-F+2P}{S}+1$$
    - Input volume size ($W$)
    - The receptive field size of the Conv Layer neurons ($F$)
    - The stride with which they are applied ($S$), - The amount of zero padding used ($P$) on the border. 

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
[blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

# Graphics
## Voxel
- A volume representation is a specific kind of implicit representation that uses a regular
3D grid of discrete values called _voxels_

## Point cloud
- Point cloud is a discrete set of data points in space 

## Point-set registration

<img src="../img/point_reg.png" width="400">

- process of finding a spatial transformation (scaling, rotation, translation, ...) that aligns two point clouds. 

## Iterative closest point (ICP)
- algorithm employed to minimize difference between two point clouds.
- ICP is often used to reconstruct 2D or 3D surfaces from different scans, to localize robots and achieve optimal path planning

## PointNet ([paper](https://arxiv.org/pdf/1612.00593.pdf))
TODO