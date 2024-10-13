from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
import random
import math
import shutil

app = Flask(__name__)
temp_path = "static/temp/"
if (not os.path.exists(temp_path)):
    os.makedirs(temp_path)
else:
    shutil.rmtree(temp_path)
    os.makedirs(temp_path)


def addNoise(img_name, salt_prob=0.1, pepper_prob=0.1):
    img_path = os.path.join(temp_path,img_name)
    img = cv2.imread(img_path)

    total_pixels = img.shape[0] * img.shape[1]

    # Add salt noise (white pixels)
    num_salt = int(salt_prob * total_pixels)
    for _ in range(num_salt):
        y_coord = random.randint(0, img.shape[0] - 1)
        x_coord = random.randint(0, img.shape[1] - 1)
        img[y_coord, x_coord] = 255  # white pixel

    # Add pepper noise (black pixels)
    num_pepper = int(pepper_prob * total_pixels)
    for _ in range(num_pepper):
        y_coord = random.randint(0, img.shape[0] - 1)
        x_coord = random.randint(0, img.shape[1] - 1)
        img[y_coord, x_coord] = 0  # black pixel

    file_name = os.path.join(temp_path,"noisy_"+img_name)
    cv2.imwrite(file_name, img)
    return file_name

def denoiseMedian(img_name):
    img_path = os.path.join(temp_path,img_name)
    img = cv2.imread(img_path)

    img = cv2.medianBlur(img, 3)

    file_name = os.path.join(temp_path,"denoisedMedian_"+img_name)
    cv2.imwrite(file_name, img)
    return file_name

def denoiseMean(img_name):
    img_path = os.path.join(temp_path,img_name)
    img = cv2.imread(img_path)

    img = cv2.blur(img,(3,3))

    file_name = os.path.join(temp_path,"denoisedMean_"+img_name)
    cv2.imwrite(file_name, img)
    return file_name

def sharpen(img_name):
    img_path = os.path.join(temp_path,img_name)
    img = cv2.imread(img_path)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    sharpened_image = cv2.filter2D(img, -1, kernel)

    file_name = os.path.join(temp_path,"sharpened_"+img_name)
    cv2.imwrite(file_name, sharpened_image)
    return file_name

def sobel(img_name):
    img_path = os.path.join(temp_path,img_name)
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    file_name = os.path.join(temp_path,"sobel_"+img_name)
    cv2.imwrite(file_name,sobelxy)
    return file_name

def cannyEdge(img_name):
    img_path = os.path.join(temp_path,img_name)
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    file_name = os.path.join(temp_path,"cannyEdge_"+img_name)
    cv2.imwrite(file_name,canny)
    return file_name

def prewitt(img_name):
    img_path = os.path.join(temp_path,img_name)
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    prewitt_kernel_x = np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]])
    prewitt_kernel_y = np.array([[-1, -1, -1],[0,  0,  0],[1,  1,  1]])

    x = cv2.filter2D(img_gray,-1,prewitt_kernel_x)
    y = cv2.filter2D(img_gray,-1,prewitt_kernel_y)
    # xy = math.sqrt(x**2 + y**2)
    xy = x + y

    file_name = os.path.join(temp_path,"prewitt_"+img_name)
    cv2.imwrite(file_name,xy)
    return file_name
    

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['file']
        image.save(os.path.join(temp_path,image.filename))
        result = [addNoise(image.filename),denoiseMedian(image.filename),denoiseMean(image.filename),
                  sharpen("denoisedMedian_"+image.filename),sharpen("denoisedMean_"+image.filename),
                  sobel(image.filename), cannyEdge(image.filename), prewitt(image.filename)]
        return render_template('index.html',original = os.path.join(temp_path,image.filename),result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000)