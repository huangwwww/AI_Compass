# AICompass

## Project
The project aims to implement an object detection system that recognizes the landing pad for drones and indicates the direction of the landing site with respect to the centre of view. The input is an aerial image with a landing pad, and the output of the system is the direction of the landing site with respect to the centroid of the image.

## Description of Overall Software Structure
The system consists of two major parts: a neural network that detects whether an image contains some portion of a landing pad, and an algorithm that takes in an aerial image and uses the output from the trained model to approximate the centroid of the landing site.

## Source of Data
Our own dataset was created. Firstly we took photos of the landing pad placed on various potential landing surface. Secondly, these raw pictures were cut into crops with the dimension (252*252*3). These crops were mannually labelled as one/zero, depending on whether there is some portion of landing pad in the image. Note that crops where the landing pad occupies less than 1/8 of the area of the whole image is still labeled as zero, because our purpose is to find the centroid of the landing pad, rather than recognizing the boundary of the item. 

![alt text](https://github.com/Cheryl-Huang/AI_Compass/blob/master/cutting.png)

## Machine Learnign Model
A CNN network (one convolutional layer) that detects whether an image contains some portion of landing pad is built.

![alt text](https://github.com/Cheryl-Huang/AI_Compass/blob/master/nn.png)

## Algorithm for Object Localization
The input image (which is an aerial image) is cut into pieces in the same dimension as what is done in the data collection process. Each piece is then passed into the model that was trained before. The results from the trained model generates a probability distribution map, and a threshold is applied to generate a cluster distribution map. The algorithm traverses through the map and find the largest piece. The centroid of this largest piece is the approximated location of the landing pad.

![alt text](https://github.com/Cheryl-Huang/AI_Compass/blob/master/cluster.png)
