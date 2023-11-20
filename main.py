import cv2 as cv

classifier = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')#https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_upperbody.xml

def detectObject(imagepath):
    inputImage = cv.imread(imagepath)
    greyscale = cv.cvtColor(inputImage,cv.COLOR_BGR2GRAY)
    print(greyscale)
    cv.imwrite('grey{0}'.format(imagepath), greyscale)
    bodies = classifier.detectMultiScale(greyscale)
    for (x, y, width, height) in bodies:
        cv.rectangle(inputImage, (x, y), (x + width, y + height), (0, 0, 255), 1)
    cv.imwrite('Output{0}'.format(imagepath), inputImage)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detectObject('RSK.jpg')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
