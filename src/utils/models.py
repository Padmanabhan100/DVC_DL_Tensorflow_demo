import tensorflow as tf
import os
import logging


def get_VGG_16_model(input_shape,model_path):
    # get the model
    model = tf.keras.applications.vgg16.VGG16(input_shape=input_shape,
                                              weights='imagenet',
                                              include_top=False)
    
    # save and return the model and log it
    model.save(model_path)
    logging.info(f"VGG16 base model saved at {model_path}")
    return model

def prepare_model(model,CLASSES,freeze_all,freeze_till,learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till>0):
        for layer in model.layers[:freeze_till]:
            layer.trainable = False

    # create a fully connected layer
    flatten = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(units=CLASSES,
                                       activation='softmax')(flatten)

    # create a model
    full_model = tf.keras.models.Model(inputs=model.inputs,outputs=prediction)

    full_model.compile(optimizer='SGD',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    logging.info('custom model is compiled and ready to be trained :D')

    return full_model