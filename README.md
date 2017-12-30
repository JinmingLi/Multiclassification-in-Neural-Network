# Multiclassification-in-Neural-Network

This project includes the codes I wrote to solve multiclassification problem with a advanced tool--neural network. I was asked to recognize the handwritten digits (from 0 to 9). 

In order to learn the parameter from neural network, I first programmed feedforward propagation to compute cost function without regularized term (because that is easy to debug) and then I added a regularized term to the cost function. Next, I programmed the backpropagation algorithm to compute the gradient for the neural network cost function under random initialization. I conducted gradient checking to verify if it has converge and then added regularization to the gradient. 

Now I have all the code necessary to learn the parameters, I then used "fmincg" another optimization to learn the good set parameter. The result was satifying with about 95.3% traning accuracy. After this, an interesting trick is that I visualized the hidden layer of neural network to understand what my neural network is learning. 
