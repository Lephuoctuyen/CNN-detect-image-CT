import pygame
import os
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.layers import Layer
import tensorflow as tf

# Define the custom ResnetBlock class
class ResnetBlock(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        return self.relu(inputs + x)

    def get_config(self):
        config = super(ResnetBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
        })
        return config

# Initialize Pygame
pygame.init()

# Set up display
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Lung Cancer Detection")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)

# Load model with custom_objects
model = load_model('model/lung_cancer_model.h5', custom_objects={'ResnetBlock': ResnetBlock})

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return prediction[0]

def upload_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    return file_path

def display_image(file_path):
    img = pygame.image.load(file_path)
    img = pygame.transform.scale(img, (400, 400))
    screen.blit(img, (200, 50))

running = True
file_path = None
result = ""

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if 350 <= x <= 450 and 500 <= y <= 550:
                file_path = upload_image()
                if file_path:
                    result = predict_image(file_path)
                    result = 'Positive for lung cancer' if result > 0.5 else 'Negative for lung cancer'

    screen.fill(white)

    # Upload Button
    pygame.draw.rect(screen, black, [350, 500, 100, 50])
    font = pygame.font.Font(None, 36)
    text = font.render('Upload', True, white)
    screen.blit(text, (360, 510))

    # Display Image
    if file_path:
        display_image(file_path)

    # Display Result
    if result:
        result_text = font.render(result, True, black)
        screen.blit(result_text, (200, 470))

    pygame.display.flip()

pygame.quit()
