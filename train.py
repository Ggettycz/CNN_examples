import numpy as np
import os, sys

os.environ["KERAS_BACKEND"] = "tensorflow"

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import tensorflow.keras as keras

directory = 'images/d2'
image_size=(100, 100)
batch_size = 50
# Load the data and split it between train and test sets
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    directory,
    #labels="inferred",
    #label_mode="int", # would be good also binary?
    #class_names=None,
    #color_mode="rgb",
    batch_size=batch_size,
    image_size=image_size,
    #shuffle=True,
    seed=1337,
    validation_split=0.1,
    subset="both",
    #interpolation="bilinear",
    #follow_links=False,
    #crop_to_aspect_ratio=False,
)

# Model parameters
num_classes = 2
#input_shape = (100, 100, 3)
input_shape = (image_size[0], image_size[1], 3)

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

#informace o modelu
model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

epochs = 5

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

model.fit(
    train_ds,
    batch_size=batch_size,
    epochs=epochs,
    #validation_split=0.2,
    callbacks=callbacks,
)

model.save("final_model.keras")

sys.exit(1)
score = model.evaluate(x_test, y_test, verbose=0)

model.save("final_model.keras")

model = keras.saving.load_model("final_model.keras")

predictions = model.predict(x_test)

print('predictions', predictions.shape)
score = float(predictions[0][0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

print('tt', predictions[0])