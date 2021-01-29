import tensorflow as tf
import matplotlib.cm as cm
from IPython.display import Image
import numpy as np
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
tf.config.experimental.set_memory_growth(gpu[1], True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)


class detection:

    def make_gradcam_heatmap(self, 
        img_array, model, last_conv_layer_name, classifier_layer_names
    ):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions

        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)

        #classifier = model.get_layer(classifier_layer_names)
        #classifier_model = tf.keras.Model(classifier.input, classifier.output)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    def superpose(self, img, heatmap):
        # We load the original image
        img = tf.keras.preprocessing.image.img_to_array(img)
        # We rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        # We create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.4 + img
        return  tf.keras.preprocessing.image.array_to_img(superimposed_img)

    def get_img_array(self, path, size):
        #preprocess image to transfrom in a array of 256,256 shape to compatible with model
        img = tf.keras.preprocessing.image.load_img(path, target_size=size)
        array = tf.keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)

        return array

    def prediction_grad_cam(self, path, model):
        #for the prepocessing of image
        img_size = (256, 256)
        img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
        preprocess_input = tf.keras.applications.xception.preprocess_input
        img_array = preprocess_input(self.get_img_array(path, img_size))
     
        
        #we selected the last convolution layers
        #for model vgg19
        last_conv_layer_name = "block5_conv4"
        classifier_layer_names = ["block5_pool","sequential"]
        #for model densnet
        #last_conv_layer_name = "conv5_block16_concat"
        #classifier_layer_names = ["relu","sequential"]

        #Generate class activation heatmap
        heatmap = self.make_gradcam_heatmap(
            img_array, model, last_conv_layer_name, classifier_layer_names
            )
        #Generate prediction of the image
        result = model.predict(img_array)
        class_names = ['Suspicion COVID', 'Negatif COVID/Pneumonie','Suspicion Pneumonie Viral']
        prediction = class_names[np.argmax(result[0])]

        #return prediction et grad_cam
        return self.superpose(img, heatmap), prediction