import numpy as np
import os 
import PIL
import PIL.Image
import tensorflow as tf

batch_size = 32
img_size = (420, 420)
split = 0.2

# load images from data classification
train_ds = tf.keras.preprocessing.image_dataset_from_directory("data", validation_split=split, subset="training", seed=428, image_size=img_size, batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory("data", validation_split=split, subset="validation", seed=428, image_size=img_size, batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 49

input_shape = img_size + (3,)

def get_model():
  base_model = tf.keras.applications.EfficientNetB0(input_shape=input_shape,
                                                    include_top=False, # <== Important!!!!
                                                    weights='imagenet')
  base_model.trainable = False
  inputs = tf.keras.Input(shape=input_shape)
  x = base_model(inputs, training=False)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  # x = tf.keras.layers.Dense(units=128, activation="relu")(x)
  # x = tf.keras.layers.Dense(units=128, activation="relu")(x)
  outputs = tf.keras.layers.Dense(units=num_classes)(x)
  model = tf.keras.Model(inputs, outputs)

  return model

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # model.compile(
    #     optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    # )

model = get_model()
unfreeze_model(model)
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                              patience=5, min_lr=0.001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20,
  # callbacks=[early_stop]
)

model.save("train16.h5", save_format="h5")
