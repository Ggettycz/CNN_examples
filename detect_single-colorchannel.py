import numpy as np
import os, sys
import cv2

os.environ["KERAS_BACKEND"] = "tensorflow"

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import tensorflow.keras as keras

model = keras.saving.load_model("final_model.keras")

image_size=(100, 100)
batch_size = 128

#color = cv2.imread("mapa3.jpg", 1)
color2 = cv2.cvtColor(cv2.imread('mapa3.jpg'), cv2.COLOR_BGR2RGB)

b = color2[:,:,0]
g = color2[:,:,1]
r = color2[:,:,2]


rows_num = 10

z = np.zeros(b.shape,'uint8') # zeros 

color = cv2.merge([g,g,g])

#maxX = color.shape[0] - color.shape[0]%rows_num
#print ('maxx', maxX)
#color = color[:maxX,:,:]


img = keras.utils.load_img("mapa4.jpg", 
                          target_size=image_size)

img_array = keras.utils.img_to_array(img)

img_array = np.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0][0])
print(f"This image is {100 * score:.5f}% car.")

color_array = np.array(np.split(color, rows_num,0)) # ma to spatne rozmery

print('img_array', type(img_array[0]), img_array.shape)
print('color_array', type(color_array[0]), color_array.shape)

predictions2 = model.predict(color_array)
for i in range(color_array.shape[0]):
    cv2.imshow('obr',color_array[i])
    score = float(predictions2[i][0])
    print(f"This image is {100 * score:.5f}% car.", color_array[i].shape)
    cv2.waitKey(0)


#print(type(img_array[0,0,0,0]))
#print(type(color_array[0,0,0,0]))

sys.exit(0)
for i in range(10):
    for j in range(10):
        print(img_array[0][i][j], ' - ', color_array[0][i][j])