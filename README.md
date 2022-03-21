# Siamese network implementation for one-shot learning

Siamese network became more popular for tasks that involve finding similarity or learning equivalence relations between objects since it requires a small amount of data for training. Siamese neural network contains two or more identical sub-networks with shared weights followed by a distance calculation layer. The vectors of input images produced by the network will be used to calculate the distance between the images to the model and learn by optimizing the loss function. This project will build two different Siamese networks using different loss functions, contrastive loss and triplet loss, on the Omniglot dataset.

## Loss functions

The main object of the contrastive loss function and triplet loss function is to measure similarity by reducing intra-variance and increasing inter-variance of classes, and allowing the model to learn the margin of separation. The network's overall architecture is almost the same; only the number of identical networks with shared weights differ depending on the loss function.

### Contrastive loss function

<img width="651" alt="스크린샷 2022-03-20 오후 10 22 34" src="https://user-images.githubusercontent.com/74476122/159161913-c2f93da0-d7c2-491d-aa6a-47b5c105e233.png">

The equation above is the contrastive loss function where Pi and Pj are input images, D(Pi, Pj) is the Euclidean distance between two inputs, m is the margin, and yij is the true label. The contrastive loss function takes two image vectors as inputs and calculates the Euclidean distance between two vectors. Based on their true binary label, the loss value would be closer to 0 if two images are from the same class or 1 if they are different. The images from different classes would have a higher loss due to the predefined margin.

### Triplet loss function

<img width="671" alt="스크린샷 2022-03-20 오후 10 24 18" src="https://user-images.githubusercontent.com/74476122/159162002-51309ad3-1559-4ed3-8657-4e4922a9a20a.png">

The equation above is the triplet loss function, where ![anchor](https://latex.codecogs.com/svg.image?f(x_{i}^{a})), ![positive](https://latex.codecogs.com/svg.image?f(x_{i}^{p})), ![negative](https://latex.codecogs.com/svg.image?f(x_{i}^{n})) represent vector representation of anchor, positive and negative images respectively, and 𝛼 is margin.

The triplet loss function takes three network inputs: anchor, positive, and negative. Anchor is an input to be compared, positive is an input belonging to the same class as the anchor, and negative is input in a different class from the anchor. The function's main purpose is to minimise the distance between the anchor and positive while maximising the distance between anchor and negative by using margin.

## Dataset

[Omniglot dataset](https://github.com/brendenlake/omniglot#:~:text=The%20Omniglot%20data%20set%20is,Turk%20by%2020%20different%20people.) from `tensorflow_datasets`

## Methods

Siamese network models were trained and their performance were evaluated using three different datasets:
1. Training split (Dataset 1)
    - Alphabet class (0~40)
2. Test split (Dataset 2)
    - Alphabet class (41~50)
3. Training and test split combined (Dataset 3)
    - Alphabet class (0~50)
    
## Results

### Constrastive loss function 

![스크린샷 2022-03-21 오후 12 59 05](https://user-images.githubusercontent.com/74476122/159199727-5a1bf227-cf54-4fd4-b9f9-f01842f2e4f4.png)

The batch size was 128 and the number of epochs were 15.

### Triplet loss function 

![스크린샷 2022-03-21 오후 1 03 45](https://user-images.githubusercontent.com/74476122/159200038-aef3c2ac-1310-44d1-bc5a-5d8299b4a757.png)

The batch size was 128 and the number of epochs were 20.

### Accuracy comparison of Siamese networks using contrastive loss function and triplet loss function

![스크린샷 2022-03-21 오후 1 04 53](https://user-images.githubusercontent.com/74476122/159200113-81fc1cb3-1958-49a1-9e4b-8ac955e4087e.png)

## Generalisation capability of Siamese network

One of the main advantages of the Siamese network is that we do not need to know the number of classes before building the model. The similarity of images is calculated based on their distance; hence, the network is generalised, and there is no further training required if the data is fluid. The results show that it could still identify the similarity of images with an accuracy of approximately 67% when the model was tested on the test split. The test split of the dataset contained alphabet classes from 41 to 50, which were not included in the training process. Therefore, it shows that the Siamese network can learn discriminative features of images without training the whole dataset and still generalise the network.





