# library importation
import cv2
import matplotlib.pyplot as plt
import numpy as np

# function to import image and show it in a window
def openIMG(filename):
    image = cv2.imread('samples/' + filename)
    cv2.imshow(filename,image)
    cv2.waitKey(0)
    return image

# function that return image's propreties: shape, size, channel's number, and type
def propIMG(img):
    print("IMAGE PROPERTIES:")
    print("type:", img.dtype)
    print("height:", img.shape[0])
    print("width:", img.shape[1])
    print("nbr of channels:", img.shape[2])
    print("size:", img.size)
    
def countOccRGB(b, g, r):
    B, G, R = dict(), dict(), dict()
    for i in range(256):
        B[i], R[i], G[i] = 0, 0, 0

    for row in b:
        for value in row: B[value] += 1
    for row in g:
        for value in row: G[value] += 1
    for row in r:
        for value in row: R[value] += 1
    
    return B, G, R

# function that plots the histogram of an image
def histIMG(img, filename):
    b, g, r = cv2.split(img)
    B, G, R = countOccRGB(b, g, r)
    plt.plot(B.keys(),B.values(),color="blue")
    plt.plot(R.keys(),R.values(),color="red")
    plt.plot(G.keys(),G.values(),color="green")
    plt.title("Histogram of the file: " + filename)
    plt.show()

# function that plots histogram of an image using calcHist() of openCV
def histIMGcv(img, filename):
    i = 0
    for col in "bgr":
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist, color=col)
        plt.title("Histogram of the file: " + filename + " (using openCV calcHist())")
        i += 1
    plt.show()

# function to compute the mean of each color in the image
def meanRBG(img):
    b, g, r = cv2.split(img)
    B, G, R = countOccRGB(b, g, r)
    m = img.size
    meanB = sum([value * key for value, key in B.items()])/m
    meanG = sum([value * key for value, key in G.items()])/m
    meanR = sum([value * key for value, key in R.items()])/m
    print("Blue mean:", meanB)
    print("Green mean:", meanG)
    print("Red mean:", meanR)

# convert image to B&W using (R+G+B)/3 formula
def convIMG(img):
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            img[x][y]= (int(img[x][y][0]) + int(img[x][y][1]) + int(img[x][y][2]))/3
    cv2.imshow("file comnverted to B&W using arithmetic mean of RGB",img)
    cv2.waitKey()

# convert image to B&W using cvtColor
def cvtIMG(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image converted to gray shade",imgGray)
    cv2.waitKey()
    return imgGray
# compute gray image mean
def grayAvr(img):
    s = [img[i][j] for i in range(img.shape[0]) for j in range(img.shape[1])]
    print(s)
    return sum(s)/img.size
