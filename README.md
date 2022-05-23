# Deep Learning Framework for Facemask Detection(CNN), Facemask Removal(GAN's), and Gender Classification(CNN)

## **ABSTRACT:**
In the era of pandemics, wearing masks becomes a norm in every corner of society. Given the security risk of recording people on surveillance cameras, but not being able to detect the person's face, this paper provides solutions to the problem by building Machine Learning models to classify and predict if people are being masked or not, and furthermore predict what they might look like under the mask and the gender of such person. 
Additionally, this technology can also be used for biometric authentication studies, such as unlocking a mobile phone while putting the face masked on. This project will be deployed on a simple web application easy for users to use.


## **COMPLETE APPLICATION ARCHITECTURE FROM USER PRESPECTIVE**

![](https://github.com/rameshavinash94/CMPE258_final_Project/blob/main/img/applicaiton_flow.png?raw=true)


## **MODELS**
**List of All Models:**

Drive link : https://drive.google.com/drive/folders/1qghXvZAM4Ob1U0AsXDR0gOOPv1WodywI?usp=sharing 

**Face Mask Detection Model:**

part-a-model.sav

**GAN Discriminator and Generator Models:**

face_D_region

face_D_whole

FaceG

MaskG

face_checkpoints

mask32_checkpoints

**Gender Classification Models:**

gender_detection.model

**NOTE:** ALL THE METRICS AND VISUALIZATION CAN BE FOUND IN INDIVIDUAL COLAB NOTEBOOKS.

## **PPT LINK:**
https://www.canva.com/design/DAE96LHPDiE/gWwlP_Xbt7sU4qlzn5SR_w/view#1

## **DEPLOYMENT LINK:**
Different Repo for CI/CD:
MLOPS CI/CD pipeline repo link : https://github.com/AbrahamKong/CMPE258_face_mask_prediction

Deployment url: https://cmpe258-face-mask-prediction-h-xxxqwcgfva-uc.a.run.app/ 

We have deployed our applicaiton in GCP.

## **MODULES:**
1) DL model for Facemask Identification.
2) GAN Trained Model for uncovering face inside mask(face mask removal).
3) Trained DL model for gender classification.
4) Perform mlops CI/CD pipeline and cloud deployment to the above approach.

### **MODULE 1 - DL model for Facemask Identification**
Coronaviruses have recently become very common, contagious, and dangerous to the entire human population.The wearing of masks in public has become very common all over the world. This module includes a method for determining whether or not a face mask is worn. For this, we used a convolutional neural network. The model's accuracy is tested using various hyper parameters and multiple people at various frames.

colab link : https://github.com/poojashreeNS/cmpe_258_GANProject/tree/main

Model and Dataset link : https://drive.google.com/drive/folders/1ZFN9LBwMMDP0j957ktkTbluk0YtRFZEJ?usp=sharing

**STEPS FOR RUNNING ONLY FACEMASK IDENTIFICATION MODULE:**

Use the cell named prediction from the above colab url.
Send the folder name to the function mask_prediction. if prediction is 0 then the face in the image contains mask, else if it is 1 then it doesn't contain mask.For now I have just added print statement and the image location for target 0 and 1, which can then be replaced by GAN for 0 and Classification model for 1.

### **MODULE 2 - GAN Model for uncovering face inside mask(face mask removal)**

The goal of this module is to eliminate mask artifacts from facial images. The problem is divided into two stages: mask object identification and image completion of the mask region that has been removed. Our model's initial stage generates binary segmentation for the mask region automatically. The mask is then removed, and the afflicted region is synthesized with precise details while maintaining the overall coherency of the facial structure. We used a GAN-based network with two discriminators for this, with one discriminator learning the global structure of the face and the other discriminator focusing learning on the deep missing region. We created a matched synthetic dataset using the publicly available CelebA dataset and evaluated it on real world photos taken from the Internet to train our model in a supervised manner. In both qualitative and quantitative terms, our model outperforms previous typical state-of-the-art techniques.

Colab: https://colab.research.google.com/drive/16WWd99idWzYVU9eI8kjc_6mgJEjMpMiG?usp=sharing

Testing dataset: https://drive.google.com/drive/folders/1ZXAfrW74PAXRM5t21sbCrYWbvPilqVZy?usp=sharing 

Training dataset : https://drive.google.com/drive/folders/1d5ykCL54XJd3BUVEfh9qD15OOfaH6F6q?usp=sharing

### **MODULE 3 - Gender Classificaiton**

Because of the variety of applications, human gender detection, which is part of facial recognition, has received a lot of attention. To implement our system, we first used image processing to apply a pre-processing technique to each image (). For feature extraction, the pre-processed image is passed through the Convolution, RELU, and Pooling layers. In the image classification section, a fully connected layer and a classifier are used. To achieve a better result, we implemented our system using various optimizers. The entire method was tested using two datasets obtained from the Kaggle website.
Using the Kaggle dataset, the experimented result shows the highest accuracy of 95%.

_**Perform Gender Classification using CNN**_

**TRAINING COLAB:**  https://github.com/rameshavinash94/Gender_Classificaiton/blob/main/Gender_Classification_Training_final.ipynb

**MODEL TESTING Colab:**  https://github.com/rameshavinash94/Gender_Classificaiton/blob/main/Gender_Classificaiton_final.ipynb

Dataset Link: _https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset_ 

**TRAINED MODEL LINK:** _[https://drive.google.com/drive/folders/1xm5oE3QRjnM8vLn-m9K9Le1mBMJCafDY?usp=sharing](https://drive.google.com/file/d/1z919uBGy7Ae070oM7Q8lNE2jcBj4GRJu/view?usp=sharing)_ 

**TFLITE MODEL LINK** _https://drive.google.com/file/d/14qh6hnyXPmEcj2Yjb83n-K83mhvk87sb/view?usp=sharing_

#### **NOTE: Kindly unzip the above trained model and use it directly(gender_detection.model)...**

**RESULTS:**

![](https://raw.githubusercontent.com/rameshavinash94/Gender_Classificaiton/main/Screen%20Shot%202022-05-20%20at%207.15.35%20PM.png)

**TESTING SNIPPET**

<img width="307" alt="Screen Shot 2022-05-20 at 7 20 12 PM" src="https://user-images.githubusercontent.com/87649563/169630700-89480d25-1c61-4f01-8007-c06d77592b34.png">

<img width="122" alt="Screen Shot 2022-05-20 at 7 20 37 PM" src="https://user-images.githubusercontent.com/87649563/169630708-4a1bcbf8-83f1-4c1c-b361-00f0a551ef6e.png">

**STEPS TO FOLLOW:**
1) INSTALL THE LIBRARIES MENTIONED IN THE REQUIREMENTS FILE
   ```
   pip install -r requirements.txt
   ```
          
2) IMPORT THE REQUIRED LIBRARIES
      ```
      import cvlib as cv
      import cv2
      import matplotlib.pyplot as plt
      from tensorflow.keras.preprocessing.image import img_to_array
      import numpy as np
      import tensorflow as tf
      ```
3) LOAD THE TRAINED MODEL - gender_detection.model
      ```
      my_model = tf.keras.models.load_model('gender_detection.model') 
      ```

4) ADD THE BELOW 2 FUNCITONS
    ```
    
    def image_preprocessing(image):
      input_image = image
      face, confidence = cv.detect_face(input_image)
      start_X, start_Y, end_X, end_Y = face[0]
      resize_image = cv2.resize(input_image[start_Y:end_Y,start_X:end_X],(96,96))
      resize_image = resize_image.astype("float")/ 255.0
      img_array = img_to_array(resize_image)
      final_image = np.expand_dims(img_array, axis=0)
      return final_image
    ```
   
    ```
    def predict(preprocessed_image):
      labels = ["Man","Woman"]
      prediction = my_model.predict(preprocessed_image)[0]
      Predicted_label = labels[np.argmax(prediction)]
      return Predicted_label
    ```

5) GET AN FACIAL IMAGE AS INPUT.
    ```
      input_image = cv2.imread('/content/cr7.png') # pass in any image and test
    ```
       
6) CALL THE PREPROCESSING FUNCTION.
      ```
      preprocessed_image = image_preprocessing(input_image)
      ```
  
7) PASS THE PREPROCESSED IMAGE TO TRAINED MODEL FOR PREDICTION

     ```
     prediction = predict(preprocessed_image)
     ```
8) Finally Print the Result
    ```
    print(prediction) # return either Male or Female
    ```
 
 #### **NOTE: If we you want to use this model in web, use the Tflite model.**

**MODULE 4**
Final Colab of all modules together: 


MLOPS CI/CD pipeline repo link : https://github.com/AbrahamKong/CMPE258_face_mask_prediction

BUILD Artifacts:  https://raw.githubusercontent.com/rameshavinash94/CMPE258_final_Project/main/gcp_buildartifacts.txt


## **DEPLOYMENT ARCHITECTURE**
![](https://raw.githubusercontent.com/rameshavinash94/Cardiovascular-Detection-using-ECG-images/main/img/Deployment_diagram.png)

**PROJECT PRESENTATION VIDEO:**
