import cv2
import numpy as np
from tensorflow.keras.models import load_model

loaded_model = load_model('models/model.h5')
def preprocess_data(X):
    np_X = np.array(X)
    normalised_X = np_X.astype('float32')/255.0
    return normalised_X

classes = ['N', 'R', 'space', 'B', 'I', 'del', 'F', 'H', 'E', 'U', 'M', 'X', 'K', 'Q', 'Y', 'S', 'G', 'A', 'O', 'T', 'V', 'Z', 'C', 'P', 'L', 'W', 'D', 'nothing', 'J']
c = input('Enter the Character: ')
loc = 'test/{}_test.jpg'.format(c)
img = cv2.imread(loc)
img = cv2.resize(img, (32, 32))
img = preprocess_data(img)
img = img.reshape((1, 32, 32, 3))
prediction = np.array(loaded_model.predict(img))
predicted = prediction.argmax()
prediction_probability = prediction[0, prediction.argmax()]
print(prediction_probability)
print(classes[predicted])