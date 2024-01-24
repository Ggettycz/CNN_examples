import numpy as np
import os, sys
import cv2
import time

params = sys.argv
if len(params)<2:
    print('chybi jmeno a cesta k souboru')
    sys.exit(1)

if not os.path.exists(params[1]):
    print(f"soubor {params[1]} neexistuje")
    sys.exit(1)

input_file_name = params[1]

avg_car_size = 60
if len(params)>2 and params[2].isnumeric():
    avg_car_size = int(params[2])

os.environ["KERAS_BACKEND"] = "tensorflow"

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import tensorflow.keras as keras

model = keras.saving.load_model("color_final_model.keras")

train_image_size_x = 100
train_image_size_y = 100
batch_size = 128


#prumerna velikost v pixelech 
# zvetsit o 10% ... abychom meli trochu okoli
around = 0.8
sample_size =  around * avg_car_size
koef = max(train_image_size_x, train_image_size_y) / sample_size
print ('working with sample size:', sample_size, koef )

# ----------- nacti obrazek ------------
img = cv2.imread(input_file_name,1)
# preved na sedou .... zkousime to 
#img  = cv2.cvtColor(cv2.imread(input_file_name,1), cv2.COLOR_BGR2GRAY)
#png",1)

list_of_predictions = list()
#sys.exit( 0)
img_size_y = 100 * (round(img.shape[1] * koef) // 100 )
img_size_x = 100 * (round(img.shape[0] * koef) // 100 )
img_size = (int(img_size_y), int(img_size_x))
cols_num = img_size_y//100 - 1 
rows_num = img_size_x//100 - 1

print('desired image size', rows_num, img.shape[0], '->', img_size_x, cols_num, img.shape[1], '->', img_size_y)

img_stretch = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)


img_cropped = img_stretch.copy()
img_dots = img_stretch.copy()

#crop_x = train_image_size[0]//2
#crop_y = train_image_size[1]//2

splitted_imgs = []
coords = []

movement = 10
circle_size = movement // 2 + 1
range_size = train_image_size_x // movement

for crx in range(range_size):
    print (' ----- starting new row:', crx)
    for cry in range(range_size):
        img_cropped = img_stretch.copy()
        crop_x = crx * movement + movement
        
        crop_y = cry * movement + movement
        #print('-crop:', crx, cry, crop_x, crop_y)
        max_crop_x = img_size[1]+crop_x-train_image_size_x
        max_crop_y = img_size[0]+crop_y-train_image_size_y
        #print('--- size to ' , crop_x, '->', max_crop_x, crop_y, '->', max_crop_y, 'rows/cols', rows_num, cols_num )
        img_cropped = img_cropped[crop_x:max_crop_x,crop_y:max_crop_y]
        color = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)

        color_array = np.array(np.split(color, rows_num,0)) # ma to spatne rozmery

        for j in range(len(color_array)):
            ca = np.split(color_array[j],cols_num, axis=1)
            for i in range(len(ca)):
                splitted_imgs.append(ca[i])
                coords.append((crx,cry,j,i))

splitted_imgs = np.array(splitted_imgs)
print(len(splitted_imgs))
print(type(splitted_imgs))
print(splitted_imgs.shape)
predictions2 = model.predict(splitted_imgs, verbose=1)
#print(predictions2[0])
#print(coords[0])

print('----- kreslim kolecka -------')
min_prediction = 0.6
index = 0
img_dots = img_stretch.copy()
dot_count = 0

for index in range(len(splitted_imgs)):
    score = predictions2[index][0]
    if score > min_prediction:
        dot_count += 1
        (crx, cry, j, i) = coords[index]
        crop_x = crx * movement + movement
        crop_y = cry * movement + movement

        orig_x = (j*train_image_size_x+crop_x)  #110, 110
        orig_y = (i*train_image_size_y+crop_y)  #205, 305

        #print(f"This image is {100 * predictions2[i][1]:.5f}% car.", coords[index])
        cv2.circle(img_dots, ( orig_y + train_image_size_y // 2,  orig_x + train_image_size_x // 2), 
                               circle_size, (0,255*score,255*score),-1)
        cv2.rectangle(img_dots, ( orig_y,  orig_x), ( orig_y + train_image_size_y,  orig_x + train_image_size_x), (127,0,0),1)
                    
cv2.imshow('obr',img_stretch)
img_dots = cv2.resize(img_dots,None,fx=1/koef, fy=1/koef, interpolation=cv2.INTER_NEAREST)
cv2.imshow('dots',img_dots)
file_name= "images/output/dots-" + str(int(time.time())) + ".png"
cv2.imwrite(file_name, img_dots )
print(f"celkem nakresleno .... {dot_count} kolecek pro {min_prediction} prediction")
cv2.waitKey(0)
cv2.destroyAllWindows()

sys.exit(0)         


