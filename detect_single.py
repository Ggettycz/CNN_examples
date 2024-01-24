import numpy as np
import os, sys
import cv2

os.environ["KERAS_BACKEND"] = "tensorflow"

params = sys.argv
if len(params)<2:
    print('chybi jmeno a cesta k souboru')
    sys.exit(1)

if not os.path.exists(params[1]):
    print(f"soubor {params[1]} neexistuje")
    sys.exit(1)

fileName = params[1]
# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import tensorflow.keras as keras

model = keras.saving.load_model("final_model.keras")
#model.summary()

image_size=(100, 100)
batch_size = 128

img = cv2.imread(fileName, 1)
img = cv2.resize(img, image_size)

#maxX = color.shape[0] - color.shape[0]%rows_num

#print ('maxx', maxX)
#color = color[:maxX,:,:]

cv2.imshow('o1', img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_array = np.expand_dims(img, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0][0])
print(f"This image is {100 * score:.5f}% car.")

cv2.imshow('o1', img)


cv2.waitKey(0)
cv2.destroyAllWindows()

sys.exit(0)
