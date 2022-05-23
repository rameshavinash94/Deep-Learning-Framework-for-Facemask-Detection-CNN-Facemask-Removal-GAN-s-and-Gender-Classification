import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
class Load_model:
    def __init__(self, mask_model, face_model, mask_checkpoint_dir, face_checkpoint_dir):
        self.mask_model = mask_model
        self.face_model = face_model
        self.mask_model.build(input_shape=(None, 128, 128, 3))
        self.face_model.build(input_shape=[(None, 128, 128, 3), (None, 128, 128, 1)])
        self.mask_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.face_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.mask_checkpoint_dir = mask_checkpoint_dir
        self.face_checkpoint_dir = face_checkpoint_dir
        self.mask_checkpoint = tf.train.Checkpoint(generator_optimizer=self.mask_optimizer,
                                            generator=self.mask_model)
        self.face_checkpoint = tf.train.Checkpoint(generator_optimizer=self.face_optimizer,
                                            generator=self.face_model)
    def load(self):
        self.mask_checkpoint.restore(tf.train.latest_checkpoint(self.mask_checkpoint_dir))
        self.face_checkpoint.restore(tf.train.latest_checkpoint(self.face_checkpoint_dir))
    

    def noise_processing(self,generate_image):
      generate_image = generate_image.numpy()
      batch, height, width  = generate_image.shape[0], generate_image.shape[1], generate_image.shape[2]
      generate_image = generate_image[:, :, :, 0]
      k = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
      for i in range(batch):
          generate_image[i]= cv2.erode(generate_image[i], k)        
          generate_image[i] = cv2.dilate(generate_image[i], k)
      generate_image = np.where(generate_image >= -0.9, 1, -1)
      generate_image = tf.convert_to_tensor(generate_image, dtype=tf.float32)
      generate_image = tf.reshape(generate_image, [batch, height, width , 1])
      return generate_image

    def one_predict(self, img):
        # img = plt.imread(img_dir)
        # if img_dir.endswith('.png'):
        #     img = img * 255.0
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img_array, [128, 128],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.reshape(img, [1, 128, 128, 3])
        img = tf.cast(img, tf.float32)
        img =  (img / 127.5) - 1
        mask = self.mask_model(img, training=False)
        face = self.face_model([img, mask], training=False)
        fig = plt.figure(figsize=(7,7))
        plt.subplot(1, 2, 1)
        plt.title('Image with mask')
        plt.imshow(img[0][:,:,::-1] * 0.5 + 0.5)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Prediction Image')
        # uncover = self.noise_processing(face)
        plt.imshow(face[0][:,:,::-1]* 0.5 + 0.5)
        plt.axis('off')
        fig.savefig("test.jpeg")
        from PIL import Image
        image = Image.open('test.jpeg')
        st.subheader("Results Post Running GAN model - uncover mask under the mask")
        st.image(image,channels="BGR",width=250)
        fig2 = plt.figure(figsize=(7,7))
        plt.imshow(face[0][:,:,::-1] * 0.5 + 0.5,cmap=plt.get_cmap("Greys"))
        fig2.savefig("unmasked.jpeg")
        image1 = Image.open('unmasked.jpeg')
        # st.image(image1,channels="BGR")
        # tensor = face*255
        # face = np.array(tensor, dtype=np.uint8)
        # if np.ndim(face)>3:
        #   assert face.shape[0] == 1
        #   face = face[0]
