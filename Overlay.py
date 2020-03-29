#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# In[ ]:


content = 'hiking.jpg'             ### ENTER IMAGE NAME HERE ###
style = 'fractal.jpg'              ### ENTER IMAGE NAME HERE ###
mask = 'hiking_mask.jpg'           ### ENTER IMAGE NAME HERE ###
background = 'background1.jpg'     ### ENTER IMAGE NAME HERE ###


# In[ ]:


content_image = plt.imread(f'content/{content}')
mask_image = plt.imread(f'content/masks/{mask}')


# In[ ]:


content_copy = content_image.copy()
content_copy[mask_image == 255] = 255


# In[ ]:


content_img = Image.fromarray(content_copy)


# In[ ]:


content_img = content_img.convert('RGBA')
data = content_img.getdata()


# In[ ]:


newData = []
for item in data:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)


# In[ ]:


content_img.putdata(newData)
content_img.save(f'content/transparent/{content}', 'PNG')


# In[ ]:


background_image = Image.open(f'backgrounds/{background}')
content_trans = Image.open(f'content/transparent/{content}')


# In[ ]:


background_image = background_image.convert('RGBA')


# In[ ]:


background_image.alpha_composite(content_trans)


# In[ ]:


background_image.save(f'pictures/{style[:-4]}_{content[:-4]}.png', 'PNG')

