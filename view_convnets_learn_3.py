import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import cv2

#Visualizing heatmpas of class activation

#from keras.applications.vgg16 import VGG16
#import matplotlib.pyplot as plt
#from keras import backend as K


DATASET_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/via-dataset/images/"
RESULT_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/Doutorado_Qualificacao/results/"


def create_model(model_type):
    
    #CNN Parameters
    IMAGE_CHANNELS=3
    POOLING = None # None, 'avg', 'max'
    
    # load model and preprocessing_function
    if model_type=='VGG16':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions  
        model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='VGG19':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='Xception':
        image_size = (299, 299)
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        model = Xception(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet50':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        model = ResNet50(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
        model = ResNet101(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
        model = ResNet152(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))    
    elif model_type=='ResNet50V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        model = ResNet50V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input
        model = ResNet101V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
        model = ResNet152V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionV3':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        model = InceptionV3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionResNetV2':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        model = InceptionResNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='MobileNet':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        model = MobileNet(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))       
    elif model_type=='DenseNet121':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        model = DenseNet121(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='DenseNet169':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
        model = DenseNet169(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,)) 
    elif model_type=='DenseNet201':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        model = DenseNet201(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetLarge':
        image_size = (331, 331)
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        model = NASNetLarge(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetMobile':
        image_size = (224, 224)
        from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
        model = NASNetMobile(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='MobileNetV2':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input        
        model = MobileNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='EfficientNetB0':
        image_size = (224, 224)
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        model = EfficientNetB0(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB1':
        image_size = (240, 240)
        from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
        model = EfficientNetB1(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB2':
        image_size = (260, 260)
        from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input
        model = EfficientNetB2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB3':
        image_size = (300, 300)
        from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
        model = EfficientNetB3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB4':
        image_size = (380, 380)
        from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
        model = EfficientNetB4(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB5':
        image_size = (456, 456)
        from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
        model = EfficientNetB5(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB6':
        image_size = (528, 528)
        from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input
        model = EfficientNetB6(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB7':
        image_size = (600, 600)
        from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
        model = EfficientNetB7(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))                
    else: print("Error: Model not implemented.")

    preprocessing_function = preprocess_input

    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model
    
    output = Flatten()(model.layers[-1].output)   
    model = Model(inputs=model.inputs, outputs=output)
        
    return model, preprocessing_function, decode_predictions, image_size

#---------------- main --------------------------------


#--- CNN model ---------
K.clear_session()

model_type = 'VGG16'
model, preprocessing_function, decode_predictions_function, image_size = create_model(model_type)

# The local path to our target image
img_path = "/home/jeffersonpasserini/dados/ProjetosPos/Doutorado-Analise-MapaCalor-Aprendizado-CNN/clear.000.jpg"

# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(224, 224))

# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Finally we preprocess the batch
# (this does channel-wise color normalization)
x = preprocessing_function(x)

preds = model.predict(x)
print('Predicted:', decode_predictions_function(preds, top=3)[0])

np.argmax(preds[0])

# This is the "african elephant" entry in the prediction vector
image_output = model.output[:, 386]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = model.get_layer('block5_conv3')

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(image_output, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

#--- OpenCV Processing Image ---------

# We use cv2 to load the original image
img = cv2.imread(img_path)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img

# Save the image to disk
cv2.imwrite("/home/jeffersonpasserini/dados/ProjetosPos/Doutorado-Analise-MapaCalor-Aprendizado-CNN/clear.000.colormap.jpg", superimposed_img)