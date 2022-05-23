import numpy as np
import streamlit as st
import tensorflow as tf
import cvlib as cv
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
import tensorflow as tf
import joblib
from loadclass import Load_model
from vgg_model import VGG19_model
import cv2

# Gender Classification
def image_preprocessing(image):
  input_image = image
  face, confidence = cv.detect_face(input_image)
  start_X, start_Y, end_X, end_Y = face[0]
  resize_image = cv2.resize(input_image[start_Y:end_Y,start_X:end_X],(96,96))
  resize_image = resize_image.astype("float")/ 255.0
  img_array = img_to_array(resize_image)
  final_image = np.expand_dims(img_array, axis=0)
  return final_image
def predict(preprocessed_image):
  my_model = tf.keras.models.load_model('gender_detection.model')
  labels = ["Man","Woman"]
  prediction = my_model.predict(preprocessed_image)[0]
  Predicted_label = labels[np.argmax(prediction)]
  return Predicted_label
def gender_classification(opencv_image):
    st.subheader("Gender Classificaiton")
    preprocessed_image = image_preprocessing(opencv_image)
    prediction = predict(preprocessed_image)
    return prediction
# Facemask Detection


def mask_predict(img):
    st.subheader("Mask Detection")
    gcs_path = 'gs://face-mask-detection-model/part-a-model.sav'
    model = joblib.load(tf.io.gfile.GFile(gcs_path, 'rb'))
    # if type(img) == str:
    #     img = cv2.imread(img)
    img = cv2.resize(img,(200,200))
    img = img / 255
    if model.predict(np.array([img]))[0] > 0.5:
        predict = 1
        predictString = "unmasked"
    else:
        predict = 0
        predictString = "masked"
    st.write("Facemask Detection: This person is: ",  predictString)
    return predict
# Facemask Removal
def facemaskRemoval(img):
      mask_g = "MaskG"
      face_g = "FaceG"
      face_D_region = "face_D_region"
      face_D_whole = "face_D_whole"
      mask_g_model = tf.keras.models.load_model(mask_g, compile=False)
      face_g_model= tf.keras.models.load_model(face_g, compile=False)
      face_D_whole_model= tf.keras.models.load_model(face_D_whole, compile=False)
      face_D_region_model= tf.keras.models.load_model(face_D_region, compile=False)
      vgg_model = VGG19_model()
      try:
        test = Load_model(mask_g_model, face_g_model, mask_checkpoint_dir="mask32_checkpoints", face_checkpoint_dir="face_checkpoints")
        test.load()
        # st.image(img,channels="BGR")
      except:
        print("some error")
      img = test.one_predict(img)
# Define the Image function
def predictImage(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    #call pooja's function/model to check if mask exist
    if (mask_predict(opencv_image) == 1):
      st.write("Gender Classification: This person is a ",  gender_classification(opencv_image))
    else:
      facemaskRemoval(opencv_image)
      removed_mask = cv2.imread('unmasked.jpeg')
      #finally call the gender classificaiton function
      st.write("Gender Classification: This person is a ",  gender_classification(removed_mask))
# Create the Application
st.title('Gender Classification, Facemask Detection, and Facemask Removal')
uploaded_file = st.file_uploader("Choose a image file", type="jpeg")
input = st.button('Predict')
# Generate the prediction based on the users input
if input:
    st.image(uploaded_file, channels="BGR",width=150)
    predictImage(uploaded_file)









