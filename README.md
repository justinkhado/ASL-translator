# CRFNN-NST-Overlay



### Step 1: 
Clone the github repository:
git clone github.com/justinkhado/neural-cycle
Download the weights for the CRF model at https://goo.gl/ciEYZi and place it in the neural-cycle directory

### Step 2: 
Select any image and place it in the neural-cycle directory, alongside the run_demo.py file.
Open the run_demo.py file with a text editor and change the input_file variable to match the name of the image you selected.
Type python run_demo.py into the linux shell in order to run the model on your selected image.
The program will output a labels.png and an output.jpg.
The labels.png represents the mask that was created by the model, and the output.jpg represents the extracted image segment from your original image file. You will need the output.jpg file as well as the original image file to complete the following steps.

### Step 3: 
Place the content image and style image files into the corresponding content and style folders. 
Open the python file or jupyter notebook file depending on your preference.
Replace the content and style variable values with the names of your content and style images, respectively.
Run the python script to produce the neural style transformation into the Backgrounds directory.

### Step 4: 
Open the overlay file and replace the variable names with the appropriate image file names.
Run the overlay script.
The final image will be uploaded into the pictures directory.

### Step 5: 
