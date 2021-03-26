#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:59:14 2021

@author: batuhan
"""
#kütüphanler eklenir
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
#görüntü okunur ve bastırılır
img = imread("/home/batuhan/Masaüstü/gs.png")
imshow(img)
#resmin boyutları bastırırlır
print(img.shape)
#görsel yeniden boyutlandırılır, görsel ve boyutları bastırılır
resized_img = resize(img, (128,64)) 
imshow(resized_img) 
print(resized_img.shape)
#Görselin HOG özelliikleri oluşturuluyor
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
fd.shape
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
#Yeniden boyutlandırılan görsel bastırılır
ax1.imshow(resized_img, cmap=plt.cm.gray) 
ax1.set_title("Orijinal")
#Görüntüyü iyileştirmek için histogramı yeniden ölçeklendirilir
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#HOG uygulanmış görsel bastırılır
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
ax2.set_title("HOG Uygulaması")
plt.show()