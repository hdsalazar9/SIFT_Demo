#!/usr/bin/env python
# coding: utf-8

# # Object Recognition SIFT Code Example

# In[1]:


import cv2
import numpy as np


# In[2]:


#Tomado de: https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
#Tomar nuevas fotos de patitos
cam = cv2.VideoCapture(0)

cv2.namedWindow("Tomando fotos de patitos")

patito_counter = 1

while True:
    ret, frame = cam.read()
    cv2.imshow("Patito", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "patito_{}.jpg".format(patito_counter)
        cv2.imwrite(img_name, frame)
        print("Patito {} tomado!".format(img_name))
        patito_counter += 1

cam.release()

cv2.destroyAllWindows()


# In[3]:


img1 = cv2.imread("patito_1.jpg", cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, (450,300), interpolation = cv2.INTER_AREA)

img2 = cv2.imread("patito_2.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2, (450,300), interpolation = cv2.INTER_AREA)


# In[4]:


#Creando el analizador de caracteristicas SIFT

sift = cv2.xfeatures2d.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)


# In[5]:


#Hacer Match de caracteristicas con Brute Force
#Se usa normType = NORM_L1, para medir correctamente la distancia entre carac.
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)

matches = bf.match(descriptors1, descriptors2)

#Acomodar ascendentemente la distancia entre las caracteristicas matcheadas
#Entre menor sea la distancia mejor es el match
matches = sorted(matches, key = lambda x:x.distance)


# In[6]:


#Crear una imagen con el match de las dos imagenes y sus respectivos keypoints
#Imagen 1 y sus keypoints, Imagen 2 y sus keypoints, Agarrar los mejores 30 matches
matching_results = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None)

#Marcar los keypoints con su tamaño y orientacion
img1 = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2 = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# In[7]:


cv2.imshow("Patito 1", img1)
cv2.imshow("Patito 2", img2)
cv2.imshow("Matching Patitos", matching_results)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




