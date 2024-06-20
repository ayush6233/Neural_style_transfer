import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Function to load and preprocess image
def load_converted_img(uploaded_file):
    max_dim = 512
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    take_max_shape = max(shape)
    scale = max_dim / take_max_shape
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Function to show image
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

# VGG19 model for extracting style and content
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# Define style and content layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Style and content extraction
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = outputs[:self.num_style_layers], outputs[self.num_style_layers:]
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

# Calculate style and content loss
def style_content_loss(outputs, style_targets, content_targets):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_layers)
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_layers)
    total_loss = style_loss + content_loss
    return total_loss

# Function to compute gram matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

# Style and content weights
style_weight = 1e-2
content_weight = 1e4

# Streamlit app
st.title("Neural Style Transfer with VGG19")

content_file = st.file_uploader("Choose a content image...", type=["jpg", "png"])
style_file = st.file_uploader("Choose a style image...", type=["jpg", "png"])

if content_file and style_file:
    content_image = load_converted_img(content_file)
    style_image = load_converted_img(style_file)
    
    st.write("### Content Image")
    st.image(content_image.numpy(), use_column_width=True)
    
    st.write("### Style Image")
    st.image(style_image.numpy(), use_column_width=True)
    
    # Extract style and content
    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    
    # Initialize generated image
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    # Define optimizer
    opt = tf.optimizers.Adam(learning_rate=0.02)
    
    # Training step
    @tf.function
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs, style_targets, content_targets)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
        return loss
    
    epochs = st.slider("Number of epochs", 1, 10, 5)
    steps_per_epoch = st.slider("Steps per epoch", 1, 10, 5)
    
    # Run optimization
    for epoch in range(5):
        for step in range(10):
            loss = train_step(generated_image)
        st.write(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
    
    st.write("### Generated Image")
    st.image(generated_image.numpy(), use_column_width=True)
