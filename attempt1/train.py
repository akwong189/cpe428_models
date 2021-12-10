import numpy as np
import cv2
import os 
import PIL
import PIL.Image
import tensorflow as tf

batch_size = 32
img_size = (420, 420)
split = 0.2

# load all of the images
train_images = []
train_classes = []
train_boxes = []

test_images = []
test_classes = []
test_boxes = []
names = []

def preprocessing():
  cars_annos_raw = scipy.io.loadmat('cars_annos.mat')
  class_names_raw = cars_annos_raw['class_names'][0]
  annos = cars_annos_raw['annotations'][0] # ('relative_im_path', 'O'), ('bbox_x1', 'O'), ('bbox_y1', 'O'), ('bbox_x2', 'O'), ('bbox_y2', 'O'), ('class', 'O'), ('test', 'O') <=== annos

  # classes = set()
  prev_name = None
  counter = -1

  for anno in annos:
    img = cv2.imread(anno[0][0])
    h, w, _ = img.shape
    resized_img = cv2.resize(img, img_size)
    class_name = class_names_raw[anno[5][0][0] - 1][0].split(' ')[0]

    if not prev_name or prev_name != class_name:
      prev_name = class_name
      counter += 1
      names.append(class_name)
    
    if anno[6][0][0]:
      test_images.append(resized_img)
      test_classes.append(counter)
      test_boxes.append(np.array([anno[1][0][0] / w, anno[2][0][0] / h, anno[3][0][0] / w, anno[4][0][0] / h]))
    else:
      train_images.append(resized_img)
      train_classes.append(counter)
      train_boxes.append(np.array([anno[1][0][0] / w, anno[2][0][0] / h, anno[3][0][0] / w, anno[4][0][0] / h]))

# # load images from data classification
# train_ds = tf.keras.preprocessing.image_dataset_from_directory("data", validation_split=split, subset="training", seed=428, image_size=img_size, batch_size=batch_size)
# val_ds = tf.keras.preprocessing.image_dataset_from_directory("data", validation_split=split, subset="validation", seed=428, image_size=img_size, batch_size=batch_size)

# class_names = names
print(names)

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
  # x = tf.keras.layers.Dropout(0.2)(x)
  # x = tf.keras.layers.Dense(units=128, activation="relu")(x)
  # x = tf.keras.layers.Dense(units=128, activation="relu")(x)

  # perform label classification
  classifier = tf.keras.layers.Dropout(0.2)(x)
  classifier = tf.keras.layers.Dense(units=num_classes, activation='softmax', name='label')(classifier)

  # perform bounding box regression
  regression = tf.keras.layers.Dense(units=128, activation="relu")(x)
  regression = tf.keras.layers.Dense(units=64, activation="relu")(regression)
  regression = tf.keras.layers.Dense(units=4, activation='sigmoid', name='bbox')(regression)

  model = tf.keras.Model(inputs, outputs=[classifier, regression])

  return model

def unfreeze_model(model):
    # unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

model = get_model()
unfreeze_model(model)

losses = {'label': 'sparse_categorical_crossentropy',
          'bbox': 'mse'}

loss_weights = {'label': 1.0,
                'bbox': 1.0}

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=losses,
              loss_weights=loss_weights,
              metrics=['accuracy'])

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                              patience=5, min_lr=0.001)

# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)

model.fit(
  train_images,
  {"label": np.array(train_classes), "bbox": np.array(train_boxes)},
  epochs=20,
  batch_size=32,
  validation_split=0.05
)

model.save("train17.h5", save_format="h5")
