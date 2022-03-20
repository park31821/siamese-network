# Siamese network model for image similarity check

Siamese network became more popular for tasks that involve finding similarity or learning equivalence relations between objects since it requires a small amount of data for training. Siamese neural network contains two or more identical sub-networks with shared weights followed by a distance calculation layer. The vectors of input images produced by the network will be used to calculate the distance between the images to the model and learn by optimizing the loss function. This project will build two different Siamese networks using different loss functions, contrastive loss and triplet loss, on the Omniglot dataset.

## Loss functions

The main object of the contrastive loss function and triplet loss function is to measure similarity by reducing intra-variance and increasing inter-variance of classes, and allowing the model to learn the margin of separation. The network's overall architecture is almost the same; only the number of identical networks with shared weights differ depending on the loss function.

### Contrastive loss function

<img width="651" alt="스크린샷 2022-03-20 오후 10 22 34" src="https://user-images.githubusercontent.com/74476122/159161913-c2f93da0-d7c2-491d-aa6a-47b5c105e233.png">

The equation above is the contrastive loss function where Pi and Pj are input images, D(Pi, Pj) is the Euclidean distance between two inputs, m is the margin, and yij is the true label. The contrastive loss function takes two image vectors as inputs and calculates the Euclidean distance between two vectors. Based on their true binary label, the loss value would be closer to 0 if two images are from the same class or 1 if they are different. The images from different classes would have a higher loss due to the predefined margin.

### Triplet loss function

<img width="671" alt="스크린샷 2022-03-20 오후 10 24 18" src="https://user-images.githubusercontent.com/74476122/159162002-51309ad3-1559-4ed3-8657-4e4922a9a20a.png">

The equation above is the triplet loss function, where ![anchor](https://latex.codecogs.com/svg.image?f(x_{i}^{a})), ![positive](https://latex.codecogs.com/svg.image?f(x_{i}^{p})), ![negative](https://latex.codecogs.com/svg.image?f(x_{i}^{n})) represent vector representation of anchor, positive and negative images respectively, and 𝛼 is margin.

