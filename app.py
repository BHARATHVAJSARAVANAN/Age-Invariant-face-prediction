import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keras
from keras import ops
import pandas as pd
import numpy as np
import json
import os
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
#------------------------------------------------
# Dataset values
IMAGE_SIZE = 384

# Model configs
patch_size = 32
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)  # input image shape
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 50
num_patches = (IMAGE_SIZE // patch_size) ** 2
projection_dim = 64
num_heads = 4
age_k = 3

# Transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4

# MLP units
mlp_head_units = [2048, 1024, 512, 64, 32]


# FaceVit losses and metrics
facevit_losses = {
    "age": keras.losses.MeanSquaredError(),
}

facevit_metrics = {
    "age": [keras.metrics.MeanAbsoluteError()],
}

def model_compiler(model, optimizer, loss, metrics):
    "Model compilation function"
    model.compile(optimizer= optimizer, loss= loss, metrics = metrics)
    return model

def mlp(x, hidden_units, dropout_rate, block_name):
    """Simple MLP with dropout"""
    for i in range(len(hidden_units)):
        x = layers.Dense(hidden_units[i], activation=keras.activations.gelu, name= f'Dense_{i}_{block_name}')(x)
        x = layers.Dropout(dropout_rate, name = f'Dropout_{i}_{block_name}')(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size, name):
        super().__init__()
        self.patch_size = patch_size
        self.name = name

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size,
                       'name': self.name})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, name):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.name = name

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded
    
class Class_Embeddings(layers.Layer):
    def __init__(self, projection_dim, name=None):
        super(Class_Embeddings, self).__init__(name=name)
        self.projection_dim = projection_dim
        self.age_cls_embedding = self.add_weight(
            shape=(1, 1, projection_dim),
            initializer='random_normal',
            trainable=True,
            name='age_cls_embedding'
        )
        

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        age_cls_embedding = tf.tile(self.age_cls_embedding, [batch_size, 1, 1])
        return age_cls_embedding



def build_facevit(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
    # age_bins_num
):
    inputs = keras.Input(shape=input_shape, name = 'Input')

    # Create patches
    patches = Patches(patch_size, name = 'Patch_creator')(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim, name ='Patch_encoder')(patches)

    # Create the class tokens for the classification tasks
    class_tokens = Class_Embeddings(projection_dim, name='Class_Encoder')
    age_cls_embedding = class_tokens(inputs)

    # Pre-pend the tokens to the encoded_patches (age )
    encoded_patches = layers.Concatenate(axis=1, name= 'embed_concat')([age_cls_embedding, encoded_patches])

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6, name = f'LayerNorm_1_block_{i}')(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name = f'MultiHeadAttn_block_{i}'
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add(name = f'Skip_1_block_{i}')([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6, name = f'LayerNorm_2_block_{i}')(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1, block_name = f'trans_block_{i}')
        # Skip connection 2.
        encoded_patches = layers.Add(name = f'Skip_2_block_{i}')([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6, name ='LayerNorm_transformed')(encoded_patches)
    representation = layers.Flatten(name = 'Flatten_transformed')(representation[:, 2:, :])
    representation = layers.Dropout(0.3, name = 'Dropout_transformed')(representation)

    # Get the transformed class/age tokens for the classifation tasks
    age_token = encoded_patches[:, 0, :]   

    # Add MLP.
    features_age = mlp(age_token, hidden_units=mlp_head_units, dropout_rate=0.3, block_name= 'MLP_age_out')

    # FaceViT output layers
    # age_classifier = layers.Dense(age_bins_num, activation= 'softmax', name= 'age') (features_age)
    age_classifier = layers.Dense(features_age, activation= 'softmax', name= 'age')
    # return keras.Model(inputs=inputs, outputs=[age_classifier], name = 'FaceVit')
    return 

# # Load the trained model
model = "checkpoints/facevit_model.h5"

model = build_facevit(
    input_shape,          # Define the input shape
    patch_size,           # Define the patch size
    num_patches,          # Define the number of patches
    projection_dim,       # Define the projection dimension
    num_heads,            # Define the number of attention heads
    transformer_units,    # Define the transformer units
    transformer_layers,   # Define the number of transformer layers
    mlp_head_units,       # Define the MLP head units
)

app.secret_key = 'your_secret_key' 

# # Directories and file paths
image_directory = 'FGNET/images'  
csv_file_path = 'FGNET/train.csv'  
age_bins_file = 'age/age_bins.json'  

# Load the CSV file
df = pd.read_csv(csv_file_path)
csv_image_data = df[['image_file', 'age']].set_index('image_file').to_dict()['age']

# Load the age bin data from JSON
with open(age_bins_file, 'r') as f:
    age_bins = json.load(f)

# Function to find the appropriate age bin for a given age
def get_age_bin(age):
    for age_range, bin_number in age_bins.items():
        age_min, age_max = map(int, age_range.split('-'))
        if age_min <= age <= age_max:
            return bin_number
    return None

# Function to resize image
def resize_image(img_path, output_size=(300, 300)):
    with Image.open(img_path) as img:
        img.thumbnail(output_size)
        buffer = BytesIO()
        img.save(buffer, format=img.format)
        return buffer.getvalue()

# Function to convert image to base64 (to display in HTML)
def image_to_base64(img_path, output_size=(300, 300)):
    img_data = resize_image(img_path, output_size)
    return base64.b64encode(img_data).decode('utf-8')

# Flask route to handle image uploads and display similar images
@app.route('/', methods=['GET', 'POST'])
def upload_and_display():
    images_with_age = []
    if request.method == 'POST':
        uploaded_image = request.files['image']
        if uploaded_image:
            filename = uploaded_image.filename
            uploaded_image.save(os.path.join(image_directory, filename))
            uploaded_digits = filename[1:3]
            filtered_images = [img for img in csv_image_data.keys() if img[1:3] == uploaded_digits]
            # Prepare image data for display
            for img_name in filtered_images:
                img_path = os.path.join(image_directory, img_name)
                if os.path.exists(img_path):
                    age = csv_image_data[img_name]  
                    age_bin = get_age_bin(age) 
                    img_base64 = image_to_base64(img_path)
                    images_with_age.append({'age': age, 'age_bin': age_bin, 'img_data': img_base64})

            # return redirect(url_for('upload_and_display'))

    return render_template('index.html', images=images_with_age)

if __name__ == '__main__':
    app.run(debug=True)
