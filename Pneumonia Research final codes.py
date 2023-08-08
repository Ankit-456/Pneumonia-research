#!/usr/bin/env python
# coding: utf-8

# # Homomorphic Filter

# In[1]:


#Getting output at runtime
import logging
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)

# Homomorphic filter class
class HomomorphicFilter:
    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0]) ** 2)))
        return (1 - H)

    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b * H) * I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):
        # Validate image
        if len(I.shape) != 2:
            raise ValueError('Improper image shape. Expected 2D image.')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'external':
            if len(H.shape) != 2:
                raise ValueError('Invalid external filter. Expected 2D filter.')
        else:
            raise ValueError('Selected filter not implemented.')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt)) - 1
        return np.uint8(I)

def main():
    # Code parameters
    path_in = ''
    path_out = ''
    img_filename = 'Normal.jpeg'

    # Derived code parameters
    img_path_in = path_in + img_filename
    img_path_out = path_out + 'filtered.png'

    # Read the image
    img = cv2.imread(img_path_in, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.error("Failed to read the image: %s", img_path_in)
        return

    # Apply homomorphic filtering
    filter_params = [30, 2]
    homo_filter = HomomorphicFilter(a=0.75, b=1.25)
    try:
        img_filtered = homo_filter.filter(I=img, filter_params=filter_params)
    except ValueError as e:
        logging.error("Error occurred during filtering: %s", str(e))
        return

    # Display the original and filtered images
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", img)

    cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Filtered Image", img_filtered)
    cv2.waitKey(0)

    

if __name__ == "__main__":
    main()


# # FAWT

# In[2]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

def fawt_decomposition(img, window_size, overlap, scales, max_iterations, epsilon):
    # Initialize FAWT coefficients
    coeffs = np.zeros((scales, img.shape[0], img.shape[1]), dtype=np.float64)

    # Apply a window function
    window = signal.windows.hann(window_size)

    # Perform FAWT for each scale
    for j in range(scales):
        # Initialize window weights and spectrum
        W = np.ones_like(img, dtype=np.float64)
        S = np.zeros_like(img, dtype=np.float64)

        # Iterate until convergence
        for _ in range(max_iterations):
            # Divide image into analysis windows
            for i in range(0, img.shape[0] - window_size + 1, overlap):
                for k in range(0, img.shape[1] - window_size + 1, overlap):
                    # Apply window weights and compute FFT
                    windowed_img = img[i:i+window_size, k:k+window_size] * window
                    fft = np.fft.fft2(windowed_img)

                    # Update spectrum
                    S[i:i+window_size, k:k+window_size] += np.abs(fft)

            # Update window weights
            W = 1 / (S + epsilon)

        # Compute FAWT coefficients
        for i in range(0, img.shape[0] - window_size + 1, overlap):
            for k in range(0, img.shape[1] - window_size + 1, overlap):
                # Apply window weights and compute FFT
                windowed_img = img[i:i+window_size, k:k+window_size] * window
                fft = np.fft.fft2(windowed_img)

                # Save normalized FFT coefficients
                coeffs[j][i:i+window_size, k:k+window_size] = np.abs(fft) / np.max(np.abs(fft))

        # Downsample image for the next scale
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

    return coeffs

# Load image
img = cv2.imread(r"D:\Vit Bhopal\Research work\Pneumonia dataset\Train\Pneumonia homomorphic output\filtered_person12_bacteria_46.jpeg", 0)  # Load grayscale image

# Set FAWT parameters
window_size =8   # Size of the analysis window
overlap = window_size // 2  # Overlap between windows
scales = 4  # Number of scales
max_iterations = 12  # Maximum number of iterations for each scale
epsilon = 0.001  # Stopping criterion for iterations

# Perform FAWT decomposition
coeffs = fawt_decomposition(img, window_size, overlap, scales, max_iterations, epsilon)

# Display the FAWT coefficients
plt.figure(figsize=(12, 4))
for j in range(scales):
    plt.subplot(1, scales, j+1)
    plt.imshow(coeffs[j], cmap='gray', vmin=0, vmax=0.3)
    plt.axis('off')
    plt.title(f"Scale {j+1}")
plt.tight_layout()
plt.show()


# # Feature Extraction
# 

# In[22]:


import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image

# Function to create a CNN model for feature extraction
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu')
    ])
    return model

# Load the CNN model
cnn_model = create_cnn_model()

# Function to extract features from an image using the CNN model
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Standard normalization
    features = cnn_model.predict(x)
    return features.squeeze()  # Remove the batch dimension

# Provide the paths to the folders containing the input images
folder1_path = r"D:\Vit Bhopal\Research work\Pneumonia dataset\Train\Normal homomorphic output"
folder2_path = r"D:\Vit Bhopal\Research work\Pneumonia dataset\Train\Pneumonia homomorphic output"

# Initialize lists to store features and file names
all_image_features = []
file_names = []

# Function to extract features from a folder and append to the lists
def extract_features_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            features = extract_features(img_path)
            all_image_features.append(features)
            file_names.append(filename)

# Extract features from the first folder
extract_features_from_folder(folder1_path)

# Extract features from the second folder
extract_features_from_folder(folder2_path)

# Convert the feature list to a numpy array
all_image_features = np.array(all_image_features)

# Print the number of features extracted and the shape of the array
print("Number of features extracted:", len(all_image_features))
print("Shape of the feature array:", all_image_features.shape)


# In[23]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Use PCA to reduce features to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(all_image_features)

# Create a scatter plot for the classification graph
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Classification Graph')
plt.show()


# # Feature extraction and classification

# In[20]:


#svm on Homomorphic images
import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the pre-trained VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False)

# Function to extract features from an image using VGG16 model
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = vgg16_model.predict(x)
    return features.flatten()  # Flatten the features into a 1D array

# Function to extract features from multiple folders representing different classes
def extract_features_from_folders(folders):
    image_features = []
    labels = []
    for class_label, folder_path in enumerate(folders):
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpeg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                features = extract_features(img_path)
                image_features.append(features)
                labels.append(class_label)  # Assign the class_label as the label for this image
    return np.array(image_features), np.array(labels)

# Example image folder paths for two classes
class1_folder = r"D:\Vit Bhopal\Research work\Pneumonia dataset\Train\Normal homomorphic output"
class2_folder = r"D:\Vit Bhopal\Research work\Pneumonia dataset\Train\Pneumonia homomorphic output"
# Extract features and labels from the two class folders
all_image_features, labels = extract_features_from_folders([class1_folder, class2_folder])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_image_features, labels, test_size=0.2, random_state=42)

# SVM classifier
svm_classifier = SVC()

# Train SVM classifier using the training data
svm_classifier.fit(X_train, y_train)

# Predict using the trained SVM classifier
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[21]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Use PCA to reduce features to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(all_image_features)

# Create a scatter plot for the classification graph
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Classification Graph')
plt.show()


# In[4]:


from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2'], yticklabels=['Class 1', 'Class 2'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[1]:


#svm on FAWT images
import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False)

# Function to extract features from an image using VGG16 model
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = vgg16_model.predict(x)
    return features.flatten()  # Flatten the features into a 1D array

# Function to extract features from multiple folders representing different classes
def extract_features_from_folders(folders):
    image_features = []
    labels = []
    for class_label, folder_path in enumerate(folders):
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpeg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                features = extract_features(img_path)
                image_features.append(features)
                labels.append(class_label)  # Assign the class_label as the label for this image
    return np.array(image_features), np.array(labels)

# Example image folder paths for two classes
class1_folder = r"D:\Vit Bhopal\Research work\Pneumonia dataset\Train\Pneumonia FAWT2"
class2_folder = r"D:\Vit Bhopal\Research work\Pneumonia dataset\Train\Normal FAWT2"
# Extract features and labels from the two class folders
all_image_features, labels = extract_features_from_folders([class1_folder, class2_folder])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_image_features, labels, test_size=0.2, random_state=42)

# SVM classifier
svm_classifier = SVC()

# Train SVM classifier using the training data
svm_classifier.fit(X_train, y_train)

# Predict using the trained SVM classifier
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[2]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importing the 3D plotting module

# Assuming "all_image_features" is your feature matrix with shape (n_samples, n_features)
# Assuming "labels" is a list or array containing the class labels for each data point

# Use PCA to reduce features to 3D for visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(all_image_features)

# Create a 3D scatter plot for the classification graph
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='viridis', edgecolors='k')

# Add labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('SVM Classification 3D Graph')

# Adding color bar legend
fig.colorbar(scatter)

plt.show()


# In[9]:


from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2'], yticklabels=['Class 1', 'Class 2'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[24]:


from tensorflow.keras.applications.vgg16 import VGG16

# Load the VGG16 model
model = VGG16(weights='imagenet')

# Print the model summary
model.summary()


# # Grad cam

# #

# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[23]:


model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

last_conv_layer_name = "block14_sepconv2_act"

# The local path to our target image
img_path =(r"D:\Vit Bhopal\Research work\Pneumonia dataset\Train\Normal homomorphic output\filtered_IM-0210-0001.jpeg")
display(Image(img_path))


# In[24]:


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
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


# In[25]:


# Prepare image
img_array = preprocess_input(get_img_array(img_path, size=img_size))

# Make model
model = model_builder(weights="imagenet")

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


# In[26]:


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


save_and_display_gradcam(img_path, heatmap)


# In[ ]:




