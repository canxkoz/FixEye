# FixEye
Health++ 2019 Project: An approach to non invasive and preventive healthcare solution for low resource conditions. This part of our project is built to prove that retinal images can be used in the diagnosis of diseases such as Diabetes and permanenert Optic Neural Damage casued by Diabetes. We made use of the Messidor Dataset in order to train our Neural Network. Special thanks to Michael D. Abramoff, MD, PhD and his extraordinary in the field of Retinal Imaging. Without the Messidor dataset, we would not be able to use another publicly available dataset, in order to achieve the 78% accuracy.

## Neural Networks
![Diagram](readme_images/Neural Net Diagram.png)
Our Neural network consists of two Filtering layers and a MaxPool Layer. A diagram of the Machine Learning algorithm can be seen on the figure above. neural_networks.py includes the steps above.

## Accuracy
![Accuracy](readme_images/Accuracy.PNG)
The Machine Learning Model has a 78% accuracy in spotting Diabetes by using the Fundus images in the Messidor Datset.

## Functions
functions.py includes various histogram equalizing methods in opencv whih allowed us to test out the various outcomes of the testruns that we had in the past.

## Settings
settings.py allows an easier platform to edit the learning process.

## Main

## Credits
This project is created with the collaboration of 4 undergraduate students.
1) [Can Koz](https://github.com/canxkoz)
2) [Derek Wu](https://github.com/derekwu1)
3) [Sanjiv Soni](https://github.com/sanjivsoni)
4) [Schahrouz Kakavand](https://github.com/schahrouz)


## Imports 
imports.py file contains all of the packaged that are necessary fir the 
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
