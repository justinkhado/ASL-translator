"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Modified code to use mask created by model to isolate the image segment identifying a person. Outputs this segment as jpg file to be pushed to neural style program
"""

import sys
sys.path.insert(1, './src')
from crfrnn_model import get_crfrnn_model_def
import util
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    input_file = 'hiking.jpg'
    mask_file = 'labels.png'
    
    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = 'crfrnn_keras_model.h5'

    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)

    img_data, img_h, img_w, size = util.get_preprocessed_image(input_file)
    probs = model.predict(img_data, verbose=False)[0]
    segmentation = util.get_label_image(probs, img_h, img_w, size)
    segmentation.save(mask_file)
    
    input_content = cv2.imread(input_file)
    mask_content = cv2.imread(mask_file)

    input_content[mask_content == 0] = 255
    print(input_content.shape)
    cv2.imwrite('output.jpg', input_content)

if __name__ == '__main__':
    main()
