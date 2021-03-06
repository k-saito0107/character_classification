import numpy as np
import cv2
from PIL import Image
import random

#data augumentation
class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        
        return img

class Resize():
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def __call__(self, img):
        img = self.pil2cv(img)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.bitwise_not(img)
        img = cv2.resize(img, (self.width, self.height))
        return img
    
    def pil2cv(self, image):
        ''' PIL型 -> OpenCV型 '''
        new_image = np.array(image, dtype=np.uint8)
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        return new_image

class Noise():
    def __init__(self, salt, papper):
        self.salt = salt
        self.papper = papper

    def __call__(self, img):
        height, width = img.shape
        salt = np.random.uniform(self.salt[0], self.salt[1])
        papper = np.random.uniform(self.papper[0], self.papper[1])
        for h in range(height):
            for w in range(width):
                g = img[h][w]
                if g > 220:
                    r = random.random()
                    if r < papper:
                        img[h][w] = 0
                    continue
                if g < 80:
                    r = random.random()
                  #print(r)
                    if r > (1- salt):
                        img[h][w] = 255
        return img

class Line_Noise():
    def __call__(self, img):
        r = random.randint(0, 5)
        height, width = img.shape
        if r == 1:
            for w in range(width):
                img[0][w] = 0
                img[1][w] = 0
        elif r == 2:
            for h in range(height):
                img[h][width-1] = 0
                img[h][width -2] = 0
        elif r == 3:
            for w in range(width):
                img[height-1][w] = 0
                img[height -2][w] = 0
        elif r == 4:
            for h in range(height):
                img[h][0] = 0
                img[h][1] = 0
        else:
            img = img
        
        return img