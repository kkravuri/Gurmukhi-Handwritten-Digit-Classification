# Gurmukhi Handwritten Digit Classification with python from scratch
## Gurmukhi Handwritten Digit Classification
Handwritten digit recognition is the ability of the computer to recognize the human handwritten digits. It is little difficult tasks to identity the pattern because the handwritten digits are not perfects and can be made in many different patterns this example, we are going to use the handwritten digits as image of digits and recognize the digit present in that image.

## Dataset
The Given train Dataset has 10 different classes, and, in each class, there are 100 images.
## Steps to implement the handwritten digit recognition
### Step 1: Import the libraries
TensorFlow, pathlib, matplotlib, numpy, pandas and tensorflow.keras libraries are used.
### Step 2: Load the dataset
At first, we imported the “os” module to interact with the operating system. Then we import “listdir()“ function from os to get access to the path. Then with the help of “os.listdir()” function first we will get all the list of class directories. by taking the length of all dirs, iterate through each directory and read one by one image in each class directory and append to images array.
### Step 3: visualize one instance of each class present in the dataset.
Implemented the code to view one image in each class. to see the pattern and understand the pattern of each digit.
### Step 4: Pre-process the data
Split the data into train and test with the test_size=0.25. The image data cannot be fed directly into the model, so we need to perform some operations and process the data to make it ready for our neural network. The dimension of the training data is (750, 32, 32). I need the data set as n * m, so I have taken transpose of the dataset.10 represents the one hot encoding.
### Step 5: Create the model:
Here I have used only one hidden layer.
Initialise parameters with random values. I have repeated Forward Propagation, softmax, cost function, Backword Propagation and for each step updated the parameters.
Implemented tanh, relu softmax, derivative of tanh and relu which we need in forword propagation functions. 
Implemented function for initialize the parameters by using random from numpy and stored them in the parameter dictionary and returned that dictionary.
 Implemented Forword propagation function:
 collected the inital paramters
 using relu and softmax function calculated a1 and a2.
 used np.dot for matrix multiplication
 defined the cost function.
  Implemented back propagation function
 used the parameter from forward cache as initial values.
 Update parameters:
 first I have taken parameter from previous and updated to new values.
### Step 6: Compile, fit and Evaluate Model
x,y , learning rate = 0.02 and iteration = 100 times are the input parameter to the model function. This model function invokes initialised parameter, forward propagation, cost function, backword propagation, update parameters functions recurrently until we complete given number of iterations. stored cost for each 10th iteration in the array and display after completion. plotted the graph cost vs iterations. model returns parameters and cost, called the model for X_train and y_train.
 
cost vs iterations

### Step 7: Accuracy
![image](https://user-images.githubusercontent.com/69726245/235208162-55364169-9ae4-451f-8511-edef93d411e7.png)
 
### Step 8: Validation 
Validated the model with validation data set, displayed the image with the predicted values. Fig.1 -Model predicted values correctly in some classes we can all zero were correctly predicted. Few ones were wrongly predicted shown in Fig.2
![image](https://user-images.githubusercontent.com/69726245/235208378-d68d558f-027d-46f4-a16f-3837e7fcb946.png)

### Step 9: Conclusion
I have implemented the Convolutional Neural Network for the classification of Gurumukhi dataset. Given dataset is divided into training and test dataset. It was taken to make predictions of handwritten digits from 0 to 9. The dataset was cleaned, scaled, and shaped. 
Using without using TensorFlow, a CNN model was created for multiclass classification with softmax and was eventually trained on the training dataset. Finally, predictions were made using the trained model.
## References
•	Neural Network python from scratch for MultiClass Classification with Softmax    https://www.youtube.com/watch?v=vtx1iwmOx10digits_recognition_neural_network - https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/1_digits_recognition/digits_recognition_neural_network.ipynb
•	YouTube- Neural Network For Handwritten Digits Classification | Deep Learning Tutorial 7 (Tensorflow2.0) -https://www.youtube.com/watch?v=iqQgED9vV7k&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=7
•	YouTube- Text Detection using Neural Networks | OPENCV Python https://www.youtube.com/watch?v=y1ZrOs9s2QA
•	Deep Learning Project – Handwritten Digit Recognition using Python -https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/
