import cv2 as cv

# Load the pre-trained Haar cascade classifier
classifier = cv.CascadeClassifier(
    'haarcascade_frontalface_alt2.xml')  # https://github.com/opencv/opencv/blob/4.x/data/haarcascades


def detectObject(imagepath):
    inputImage = cv.imread(imagepath)
    greyscale = cv.cvtColor(inputImage, cv.COLOR_BGR2GRAY)
    cv.imwrite('grey{0}'.format(imagepath), greyscale)
    bodies = classifier.detectMultiScale(greyscale)  # generates integral image and runs cascade Classifier on image
    # put's red box around faces
    for (x, y, width, height) in bodies:
        cv.rectangle(inputImage, (x, y), (x + width, y + height), (0, 0, 255), 1)
    # writes output image
    cv.imwrite('Output{0}'.format(imagepath), inputImage)
    return inputImage


def detectFacesFromWebcam():
    # Initialize webcam
    cap = cv.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to grayscale
        greyscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces
        faces = classifier.detectMultiScale(greyscale)

        # Draw rectangle around the faces
        for (x, y, width, height) in faces:
            cv.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

        # Display the resulting frame
        cv.imshow('Webcam Face Detection', frame)

        # Break the loop on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Show detection results on static images
    #cv.imshow('output', detectObject('download.jpeg'))
    #cv.waitKey(0)
    #cv.imshow('output', detectObject('oscars.jpg'))
    #cv.waitKey(0)

    # Run the webcam face detection
    detectFacesFromWebcam()
