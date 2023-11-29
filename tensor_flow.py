from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from PIL import Image
import smtplib
import ssl
from email.message import EmailMessage
import imghdr



from matplotlib import pyplot as plt

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0

subject = "Security Purpose"
body = "Hello. The wrong person try to access your account."
sender_email = "mandrone987@gmail.com"
receiver_email = "mdwills15@gmail.com"
password = input("Type your password and press enter:")

message = EmailMessage()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = subject
message["Bcc"] = receiver_email
message.set_content(body)

brightness = 10
contrast = 2.3

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    skip += 0.25

    # Predicts the model
    if skip % 5 == 0:
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
    # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        image = np.array(image)
        image = image.reshape((image.shape[1], -1))
        np.save("MarcusJ",image)
        picture = np.load("MarcusJ.npy")
        im = Image.fromarray(picture)
        im.convert("L")
        plt.figure(figsize=(10,10))
        plt.imshow(im)
        plt.axis("off")
        plt.savefig("MarcusP" + ".png")
        plt.show()

        with open("MarcusP.png", 'rb') as file:
            send_picture = file.read()

        if np.round(confidence_score) < 80:
            message.add_attachment(send_picture, maintype='image', subtype=imghdr.what(None, send_picture))
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
                server.send_message(message)
                server.quit()


    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
