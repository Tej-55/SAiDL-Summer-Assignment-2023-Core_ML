# Variations of Softmax

The softmax function is a commonly used activation function in neural networks, particularly for classification tasks. It allows us to map probabilistic distributions using neural networks by transforming a vector of K real numbers into a probability distribution of K possible outcomes. However, the softmax function can become computationally expensive when the number of classes is large, causing slower training and evaluation times.

## Problem Statement

In this project, my goal is to explore different variations of softmax implementations in neural networks and evaluate their impact on model performance and training time. Specifically, I will develop a convolutional neural network (CNN) model on the CIFAR 100 dataset for image classification. I will compare the standard softmax implementation with alternative softmax functions that reduce the computational complexity.

## Project Tasks

1. **Dataset**: I have used the CIFAR 100 dataset, which consists of 60,000 32x32 color images across 100 classes. The dataset is divided into 50,000 training images and 10,000 test images.

2. **Model Development**: I have designed a CNN model with a specific architecture for image classification. The initial model used the standard softmax implementation as the activation function.

3. **Alternative Softmax Implementations**: I have created a second model with the same architecture as the initial model, but with different softmax implementations that reduce the computational complexity. Examples of alternative softmax implementations include Gumbel-Softmax, and Heirarchical Softmax. I have reported the time complexity for each of the softmax functions used.

4. **Model Evaluation**: I have evaluated the performance of both models using a range of evaluation metrics, including accuracy, precision, recall, F1 score, and confusion matrix. This allow us to compare the performance of the models with different softmax implementations.

5. **Performance and Epoch Time Comparison**: I have compared the performance and epoch time between the models with different softmax implementations. This analysis will help us understand the trade-offs between model accuracy and computational complexity.

## Conclusion

By conducting this experiment, we aim to gain insights into the trade-offs between different softmax implementations in neural networks, particularly in the context of classification tasks with a large number of classes, which in this case were 100. Understanding the impact of softmax implementations on model performance and training time will enable us to make more informed decisions when designing and training neural networks for classification tasks.

In the results, it can be found that the Normal Softmax gave _slightly_ better performance on the metrics (F1 score: _0.3802_ in regular Softmax vs _0.3781_ in Gumbel), however the time required by the Gumbel Softmax was relatively lesser as compared with the regular Softmax (Mean epochal time: _6.487_ s in regular Softmax vs _5.845_ s in Gumbel). Both regular and Gumbel Softmax have the same time complexity in terms of number of classes _n_, given by O(n). However, the Gumbel Softmax can be computationally faster during training due to its reparametrization trick, which enables efficient gradient computation and backpropagation through the relaxation process. This makes it easier to optimize models that involve discrete decisions, such as reinforcement learning agents with discrete action spaces.

**Please note**, even though the implementation of Hierarchical Softmax was part of this assignment, I was unable to get it to train properly in the TensorFlow environment. I have still left the section for it in the colab notebook, as I was unable to get the problem fixed.
