# Digit Recognizer  
This is a well-known Kaggle project which is considered to be the "Hello World" of computer vision.  The data set (MNIST) is also provided on the [Kaggle platform](https://www.kaggle.com/c/digit-recognizer).  The data set contains a training data of 42000 hand-written 28 by 28 grey scale images along with the target label (0 to 9).    
## Neural Network Architecture
In this project, I used a convolutional neural network consisting of 4 2D convolutional layers and 2 fully-connected layers along with Dropout regularization.  RMSProp is used as an optimizer along with callbacks setting, which observes the validation accuracy over the epochs and if no improvement is observed over the last 3 epochs, the learning rate is made half.    

To avoid overfitting, data augmentation (rotation, width shift, height shift, shear and zoom) is used. 

The neural network is trained for 30 epochs and it resulted in an accuracy of 0.9946 on the public leaderboard of Kaggle. 
