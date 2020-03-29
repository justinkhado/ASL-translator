#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg19


# In[ ]:


content = 'hiking.jpg'    ### ENTER IMAGE NAME HERE ###
style = 'fractal.jpg'     ### ENTER IMAGE NAME HERE ###


# In[ ]:


content_image = plt.imread(f'content/{content}')
style_image = plt.imread(f'style/{style}')


# In[ ]:


h, w, _ = content_image.shape


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))
ax1.imshow(content_image)
ax1.set_title('Content Image')
ax2.imshow(style_image)
ax2.set_title('Style Image')
plt.show()


# In[ ]:


def load_image(image):
    image = plt.imread(image)
    img = tf.image.convert_image_dtype(image, tf.float32)
    img = tf.image.resize(img, [400, 400])
    img = img[tf.newaxis, :]
    return img


# In[ ]:


content_image = load_image(f'content/{content}')
style_image = load_image(f'style/{style}')


# In[ ]:


vgg_model = tf.keras.applications.VGG19(include_top=False)
vgg_model.trainable = False


# In[ ]:


content_layers = ['block4_conv2']

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']


# In[ ]:


num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# In[ ]:


def helper_model(layers, model):
    outputs = [model.get_layer(layer).output for layer in layers]
    model = Model([vgg_model.input], outputs)
    
    return model


# In[ ]:


def gram_matrix(input_tensor):
    tensor = input_tensor
    tensor = tf.squeeze(tensor)
    fun = tf.reshape(tensor, [tensor.shape[2], tensor.shape[0]*tensor.shape[1]])
    result = tf.matmul(tensor, tensor, transpose_b=True)
    gram = tf.expand_dims(result, axis=0)
    return gram


# In[ ]:


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = helper_model(style_layers + content_layers, vgg_model)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        
    def call(self, inputs):
        inputs = inputs * 255.0    # pixel values
        preprocessed_input = vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        
        return {'content': content_dict, 'style': style_dict}


# In[ ]:


extractor = StyleContentModel(style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']


# In[ ]:


optimizer = tf.optimizers.Adam(learning_rate=0.001)


# In[ ]:


style_weight = 1e-4
content_weight = 10

style_weights = {'block1_conv1': 1.0,
                 'block2_conv1': 0.8,
                 'block3_conv1': 0.5,
                 'block4_conv1': 0.3,
                 'block5_conv1': 0.1}


# In[ ]:


def loss_function(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    style_loss = tf.add_n([style_weights[name]*tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])    
    style_loss *= style_weight / num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    
    loss = style_loss + content_loss
    
    return loss


# In[ ]:


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = loss_function(outputs)
    
    gradient = tape.gradient(loss, image)
    optimizer.apply_gradients([(gradient, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


# In[ ]:


target_image = tf.Variable(content_image)


# In[ ]:


epochs = 10
steps_per_epoch = 100
step = 0
images = []

for epoch in range(epochs):
    for curr_batch in range(steps_per_epoch):
        step += 1
        train_step(target_image)    
    plt.imshow(np.squeeze(target_image.read_value(), 0))
    plt.title(f'Train step: {step}')
    images.append(np.array(tf.image.resize(target_image[0], [h,w])))
    plt.show()


# In[ ]:


for index, image in enumerate(images):
    plt.imsave(f'backgrounds/background{index+1}.jpg', image)

