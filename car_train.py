import scipy.io
import cv2
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LOAD_NEW = False

# if loading from new dataset without csv, load the mat file and parse outputting the csv and a dataframe
if LOAD_NEW:
    # load mat file for labels
    cars_annos_raw = scipy.io.loadmat('cars_annos.mat')
    class_names_raw = cars_annos_raw['class_names'][0]
    # ('relative_im_path', 'O'), ('bbox_x1', 'O'), ('bbox_y1', 'O'), ('bbox_x2', 'O'), ('bbox_y2', 'O'), ('class', 'O'), ('test', 'O') <=== annos
    annos = cars_annos_raw['annotations'][0]
    
    df = pd.DataFrame(columns=['img_path', 'label', 'x1', 'y1', 'x2', 'y2'])
    curr = []
    i = -1

    for anno in annos:
        path = anno[0][0]
        class_name = class_names_raw[anno[5][0][0] - 1][0].split(' ')[0]

        h, w, _ = cv2.imread(path).shape

        if len(curr) == 0 or class_name != curr[i]:
            i += 1
            curr.append(class_name)

        x1 = anno[1][0][0] / w
        y1 = anno[2][0][0] / h
        x2 = anno[3][0][0] / w
        y2 = anno[4][0][0] / h
        data = {"img_path": path, "label": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        df = df.append(data, ignore_index=True)

    print(curr)
    df.to_csv("car_ims.csv")
else:
    df = pd.read_csv("car_ims.csv")

# split the dataset into training and testing data
train_df, test_df = train_test_split(df, test_size=0.2)

# split training dataset into 3 arrays: image paths, labels, and bounding box
train_image_names = train_df.img_path.values
train_labels = train_df.label.values.astype(np.float32)
train_bbox = train_df[['x1', 'y1', 'x2', 'y2']].values.astype(np.float32)

# split testing dataset into 3 arrays: image paths, labels, and bounding box
test_image_names = test_df.img_path.values
test_labels = test_df.label.values.astype(np.float32)
test_bbox = test_df[['x1', 'y1', 'x2', 'y2']].values.astype(np.float32)

print(len(set(train_df.label.values)), len(set(test_df.label.values)))

# auto load into the GPU for preprocessing (faster than previous method and allows for larger images)
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = (720, 720)

@tf.function
def preprocess(image_name, label, bbox):
    image = tf.io.read_file(image_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=IMG_SIZE)

    return image, {'label': label, 'bbox': bbox}

trainloader = tf.data.Dataset.from_tensor_slices((train_image_names, train_labels, train_bbox))
testloader = tf.data.Dataset.from_tensor_slices((test_image_names, test_labels, test_bbox))

trainloader = (
    trainloader
    .map(preprocess, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

testloader = (
    testloader
    .map(preprocess, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# develop the model
num_classes = 49
input_shape = IMG_SIZE + (3,)

def get_model():
    base_model = tf.keras.applications.EfficientNetB2(input_shape=input_shape,
                                                        include_top=False,
                                                        weights='imagenet')
    base_model.trainable = False
    unfreeze_model(base_model)
        
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x) # dropout may not be needed

    # perform label classification
    # add more dense nets here
    # classifier = tf.keras.layers.Dense(units=512, activation="relu")(x)
    classifier = tf.keras.layers.Dense(units=256, activation="relu")(x)
    classifier = tf.keras.layers.Dropout(0.2)(classifier) # dropout may not be needed
    classifier = tf.keras.layers.Dense(units=128, activation="relu")(classifier)
    classifier = tf.keras.layers.Dropout(0.2)(classifier) # dropout may not be needed
    classifier = tf.keras.layers.Dense(units=49, activation='softmax', name='label')(classifier)

    # perform bounding box regression
    regression = tf.keras.layers.Dense(units=256, activation="relu")(x)
    # regression = tf.keras.layers.Dropout(0.5)(regression) # dropout may not be needed
    regression = tf.keras.layers.Dense(units=128, activation="relu")(regression)
    # regression = tf.keras.layers.Dropout(0.5)(regression) # dropout may not be needed
    regression = tf.keras.layers.Dense(units=4, activation='sigmoid', name='bbox')(regression)

    model = tf.keras.Model(inputs, outputs=[classifier, regression])

    return model

def unfreeze_model(model):
    # unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

tf.keras.backend.clear_session()
model = get_model()
# model.summary()

losses = {'label': 'sparse_categorical_crossentropy',
          'bbox': 'mse'}

metrics = {'label': "accuracy",
           'bbox': "mae"}

loss_weights = {'label': 1.0,
                'bbox': 1.0}

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=losses,
              loss_weights=loss_weights,
              metrics=metrics)
              #metrics=['accuracy', tf.keras.metrics.MeanAbsoluteError()])

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                              patience=5, min_lr=0.001)

# early stop model if label loss stops improving
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_label_loss', mode='min', patience=10, restore_best_weights=True)

model.fit(
  trainloader,
  validation_data=testloader,
  epochs=100,
  callbacks=[early_stop],
)

model.evaluate(testloader)

model.save("train25.h5", save_format="h5")
