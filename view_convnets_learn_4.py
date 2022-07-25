#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 10:05:22 2022

@author: jeffersonpasserini

https://keras.io/examples/vision/grad_cam/
"""

import numpy as np
import tensorflow as tf
import os
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))
    

def create_model(model_type):
    
    #CNN Parameters
    IMAGE_CHANNELS=3
    POOLING = None # None, 'avg', 'max'
    
    # load model and preprocessing_function
    if model_type=='VGG16':
        image_size = (224, 224)
        last_conv_layer_name = "block5_conv3"
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions   
        #model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
        model = VGG16(weights='imagenet')
    elif model_type=='VGG19':
        image_size = (224, 224)
        last_conv_layer_name = "block5_conv4"
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions   
        #model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
        model = VGG19(weights='imagenet')
    elif model_type=='Xception':
        image_size = (299, 299)
        last_conv_layer_name = "block14_sepconv2_act"
        from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions  
        #model = Xception(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
        model = Xception(weights='imagenet')
    elif model_type=='ResNet50':
        image_size = (224, 224)
        last_conv_layer_name = "conv5_block3_out"
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input, decode_predictions   
        #model = ResNet50(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
        model = ResNet50(weights='imagenet')
    elif model_type=='ResNet101':
        image_size = (224, 224)
        last_conv_layer_name = "conv5_block3_out"
        from tensorflow.keras.applications.resnet import ResNet101, preprocess_input, decode_predictions   
        #model = ResNet101(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
        model = ResNet101(weights='imagenet')        
    elif model_type=='ResNet152':
        image_size = (224, 224)
        last_conv_layer_name = "conv5_block3_out"
        from tensorflow.keras.applications.resnet import ResNet152, preprocess_input, decode_predictions   
        #model = ResNet152(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))    
        model = ResNet152(weights='imagenet')    
    elif model_type=='ResNet50V2':
        image_size = (224, 224)
        last_conv_layer_name = "conv5_block3_out"
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions   
        #model = ResNet50V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
        model = ResNet50V2(weights='imagenet')
    elif model_type=='ResNet101V2':
        image_size = (224, 224)
        last_conv_layer_name = "conv5_block3_out"
        from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input, decode_predictions   
        #model = ResNet101V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
        model = ResNet101V2(weights='imagenet')        
    elif model_type=='ResNet152V2':
        image_size = (224, 224)
        last_conv_layer_name = "conv5_block3_out"
        from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input, decode_predictions   
        #model = ResNet152V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
        model = ResNet152V2(weights='imagenet')  
    elif model_type=='InceptionV3':
        image_size = (299, 299)
        last_conv_layer_name = "mixed10"
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions   
        #model = InceptionV3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
        model = InceptionV3(weights='imagenet')  
    elif model_type=='InceptionResNetV2':
        image_size = (299, 299)
        last_conv_layer_name = "conv_7b_ac"
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions   
        #model = InceptionResNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
        model = InceptionResNetV2(weights='imagenet')  
    elif model_type=='MobileNet':
        image_size = (224, 224)
        last_conv_layer_name = "conv_pw_13_relu"
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions   
        #model = MobileNet(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))       
        model = MobileNet(weights='imagenet')       
    elif model_type=='DenseNet121':
        image_size = (224, 224)
        last_conv_layer_name = "relu"
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions   
        #model = DenseNet121(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
        model = DenseNet121(weights='imagenet')   
    elif model_type=='DenseNet169':
        image_size = (224, 224)
        last_conv_layer_name = "relu"
        from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input, decode_predictions   
        #model = DenseNet169(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,)) 
        model = DenseNet169(weights='imagenet') 
    elif model_type=='DenseNet201':
        image_size = (224, 224)
        last_conv_layer_name = "relu"
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input, decode_predictions   
        #model = DenseNet201(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
        model = DenseNet201(weights='imagenet')         
    elif model_type=='NASNetLarge':
        image_size = (331, 331)
        last_conv_layer_name = "normal_concat_18"
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input, decode_predictions   
        #model = NASNetLarge(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
        model = NASNetLarge(weights='imagenet')         
    elif model_type=='NASNetMobile':
        image_size = (224, 224)
        last_conv_layer_name = "normal_concat_12"
        from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input, decode_predictions   
        #model = NASNetMobile(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
        model = NASNetMobile(weights='imagenet')   
    elif model_type=='MobileNetV2':
        image_size = (224, 224)
        last_conv_layer_name = "out_relu"
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions         
        #model = MobileNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
        model = MobileNetV2(weights='imagenet')
    elif model_type=='EfficientNetB0':
        image_size = (224, 224)
        last_conv_layer_name = "top_activation"
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions   
        #model = EfficientNetB0(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
        model = EfficientNetB0(weights='imagenet')        
    elif model_type=='EfficientNetB1':
        image_size = (240, 240)
        last_conv_layer_name = "top_activation"
        from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input, decode_predictions   
        #model = EfficientNetB1(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
        model = EfficientNetB1(weights='imagenet')        
    elif model_type=='EfficientNetB2':
        image_size = (260, 260)
        last_conv_layer_name = "top_activation"
        from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input, decode_predictions   
        #model = EfficientNetB2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
        model = EfficientNetB2(weights='imagenet')        
    elif model_type=='EfficientNetB3':
        image_size = (300, 300)
        last_conv_layer_name = "top_activation"
        from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input, decode_predictions   
        #model = EfficientNetB3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
        model = EfficientNetB3(weights='imagenet')        
    elif model_type=='EfficientNetB4':
        image_size = (380, 380)
        last_conv_layer_name = "top_activation"
        from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input, decode_predictions   
        #model = EfficientNetB4(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
        model = EfficientNetB4(weights='imagenet')        
    elif model_type=='EfficientNetB5':
        image_size = (456, 456)
        last_conv_layer_name = "top_activation"
        from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input, decode_predictions   
        #model = EfficientNetB5(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
        model = EfficientNetB5(weights='imagenet')
    elif model_type=='EfficientNetB6':
        image_size = (528, 528)
        last_conv_layer_name = "top_activation"
        from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input, decode_predictions   
        #model = EfficientNetB6(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
        model = EfficientNetB6(weights='imagenet')        
    elif model_type=='EfficientNetB7':
        image_size = (600, 600)
        last_conv_layer_name = "top_activation"
        from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions   
        #model = EfficientNetB7(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))                
        model = EfficientNetB7(weights='imagenet')                
    else: print("Error: Model not implemented.")

    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model
    
    output = Flatten()(model.layers[-1].output)   
    model = Model(inputs=model.inputs, outputs=output)
        
    return model, preprocess_input, decode_predictions, last_conv_layer_name, image_size



#------- main -------
DATASET_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/via-dataset/images/"
RESULT_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/Doutorado-Analise-MapaCalor-Aprendizado-CNN/results/"
filenames = os.listdir(DATASET_PATH)

model_name = 'EfficientNetB0'
model, preprocess_input, decode_predictions, last_conv_layer_name, img_size = create_model(model_name)

for img_name in filenames:
    
    # The local path to our target image
    img_path = DATASET_PATH+img_name
    
    #split filename
    category = img_name.split('.')
    #mount gradcam filename
    gradcam_filename = RESULT_PATH+"gradcam/"+category[0]+"."+category[1]+"."+model_name+"."+category[2]
    
    
    print(img_path)
    
    
    # Prepare image
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    
    # Make model
    #model = model_builder(weights="imagenet")
    
    # Remove last layer's softmax
    model.layers[-1].activation = None
    
    # Print what the top predicted class is
    preds = model.predict(img_array)
    print("Predicted:", decode_predictions(preds, top=1)[0])
    
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Display heatmap
    plt.matshow(heatmap)
    plt.show()
    
    save_and_display_gradcam(img_path, heatmap, cam_path=gradcam_filename)