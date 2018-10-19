## Out Of Distribution sample detection

https://arxiv.org/abs/1802.04865

DeVries and Taylor propose to use a jointly learned confidence scalar to guide neural network learning.
A consequence of this scalar is repurposement at inference time to determine if the net is likely to need a "hint".
We interpret a high hint values (low confidence) as the image coming from outside the training distribution, and probably use this information to take special action for this sample.

------
### Approach

- During training, for each batch produce softmax probabilities as usual.
- Apply linear interploation between the given softmax and the given categorical target, with the degree of mixing determined by the confidence.
- Constrain the confidence by adding a log penalty, pulling confidence towards 1. (This introduces a dynamic min-max between classifier log-likelihood and confidence penalty, balanced with hyperparameter `lambda`).
- Once trained, test the model with in-distribution, held out data from the same domain, and data from an entirely different domain.

#### Details
- Use a "budget" moving hyper parameter to scale `lambda` and pull confidence loss towards B. Use B between 0.1 and 1.0.
- Naiive implementation gives a hint every time the network asks, killing the possibility for error to propagate from odd, but valid, samples. To adjust, impose a binomial probability to give the hint, or not. The authors used p=0.5 for giving hints.


#### Roadmap
1. Implement neural network class with confidence base class (current)
2. Feed forward, fully connected case
3. Convolutional net case
4. Multiple Instance classification + regression case


#### Environment
```
tensorflow-gpu=1.11.0
keras
numpy
seaborn
```

![ood](assets/ood-alien-from-doctor-who-coloring-page.jpg)
