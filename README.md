# Deep Learning Framework for Facemask Detection(CNN), Facemask Removal(GAN's), and Gender Classification(CNN)

## **ABSTRACT:**


## **PPT LINK:**
https://www.canva.com/design/DAE96LHPDiE/gWwlP_Xbt7sU4qlzn5SR_w/view#1

## **DEPLOYMENT LINK:**

We have deployed our applicaiton in GCP.

## **MODULES:**
1) DL model for Facemask Identification.
2) GAN Trained Model for uncovering face inside mask(face mask removal).
3) Trained DL model for gender classification.
4) Perform mlops CI/CD pipeline and cloud deployment to the above approach.

### **MODULE 1**

### **MODULE 2**

### **MODULE 3 - Gender - Classificaiton**

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
MLOPS CI/CD pipeline repo link : https://github.com/AbrahamKong/CMPE258_face_mask_prediction
BUILD Artifacts:

## **COMPLETE APPLICATION ARCHITECTURE FROM USER PRESPECTIVE**

![](https://github.com/rameshavinash94/CMPE258_final_Project/blob/main/img/applicaiton_flow.png?raw=true)

## **DEPLOYMENT ARCHITECTURE**
![](https://raw.githubusercontent.com/rameshavinash94/Cardiovascular-Detection-using-ECG-images/main/img/Deployment_diagram.png)

## **SAMPLE DEMO GIF**

![]()

**PROJECT PRESENTATION VIDEO:**
