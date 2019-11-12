# FixEye

health++ Stanford’s Health Hackathon 2019 project: Low resource areas lack preventive and screening services, which causes millions of deaths per year. Our mission is to detect chronic diseases through non-invasive retinal scans. We trained a convolutional neural network in order to determine if someone has a chance of having Diabetic Retinopathy. We used the Messidor dataset to train our model. Special thanks to Michael D. Abramoff, MD, PhD, who is an expert in the field of Retinal Imaging, for making this dataset publicly available. By using his dataset, we are able to determine if a person has the possibility of having Diabetic Retinopathy with an accuracy of 78%.

## Neural Networks
![Diagram](readme_images/Neural_Net_Diagram.png)

Our neural network consists of two filtering layers and a MaxPool layer. A diagram of the Machine Learning algorithm can be seen above. neural_networks.py includes the steps above.

## Accuracy
![Accuracy](readme_images/Accuracy.PNG)
The Machine Learning Model has a 78% accuracy in spotting Diabetes by using the Fundus images in the Messidor Datset.

## Functions
functions.py includes various histogram equalizing methods in opencv which allowed us to test out the various outcomes of the test runs that we decided while we were starting to work on the solution.

## Settings
settings.py allows an easier platform to edit the learning process.

## Main
Imports all of the other functions and trains the Comvolutional Neural Network. 

## More Info
More information about the aim of our project can be found in the presentation and on [Devpost](https://devpost.com/software/fixeye)

## Credits
This project is created with the collaboration of:
1) [Can Koz](https://github.com/canxkoz)
2) [Sanjiv Soni](https://github.com/sanjivsoni)
3) [Schahrouz Kakavand](https://github.com/schahrouz)

## Imports 
imports.py file contains all of the packaged that are necessary for the 
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
