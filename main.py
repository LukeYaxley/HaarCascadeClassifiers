import cv2 as cv
#trains cascade of classifiers on positive and negative examples of images containing faces using Adaboost
classifier = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')#https://github.com/opencv/opencv/blob/4.x/data/haarcascades


def detectObject(imagepath):
    inputImage = cv.imread(imagepath)
    greyscale = cv.cvtColor(inputImage,cv.COLOR_BGR2GRAY)
    print(greyscale)
    cv.imwrite('grey{0}'.format(imagepath), greyscale)
    bodies = classifier.detectMultiScale(greyscale) #generates integral image and runs cascade Classifier on image
    # put's red box around faces
    for (x, y, width, height) in bodies:
        cv.rectangle(inputImage, (x, y), (x + width, y + height), (0, 0, 255), 1)
    #writes output image
    cv.imwrite('Output{0}'.format(imagepath), inputImage)
    return inputImage
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cv.imshow('output',detectObject('download.jpeg'))
    cv.waitKey(0)
    cv.imshow('output',detectObject('oscars.jpg'))
    cv.waitKey(0)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
