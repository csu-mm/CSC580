'''
MS - Artificial Intelligence and Machine Learning
Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone
Module 1: Critical Thinking Assignment
Professor: Dr. Brian Holbert
Created by Mukul Mondal
May 12, 2026

Problem statement: 
Option #1: 
For this assignment, you will write Python code to detect faces in an image. 
Use the following Python code as a starter for your program.

Supply your own image file that contains one or more faces to identify. 
An example output for your program should be something like the following picture 
with red boxes (instead of green) drawn around each individual’s face:

import PIL.ImageDraw
import face_recognition

# Load the jpg file into a numpy array
# Find all the faces in the image 
# Use the following Python pseudocode as guidance for your solution.
numberOfFaces = len(faceLocations)
print("Found {} face(s) in this picture.".format(numberOfFaces))
# Load the image into a Python Image Library object so that you can draw on top of it and display it
pilImage = PIL.Image.fromarray(image)
for faceLocation in faceLocations:
     # Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.
     print("A face is located at pixel location Top: {}, Left {},Bottom: {}, Right: {}".format(top, left, bottom, right))
     # Draw a box around the face     
     drawHandle = PIL.ImageDraw.Draw(pilImage)     
     drawHandle.rectangle([left, top, right, bottom], outline="red")
# Display the image on screenpilImage.show()

Develop the remaining code in the section specific to face detection. 
Because most human faces have roughly the same structure, the pre-trained 
face detection model will work well for almost any image. There's no need 
to train a new one from scratch. Use PIL, which is the Python Image Library.

Submit your image file input and completed Python source as a zip file named:

CSC580_CTA_1_1_last_name_first_name.zip.
'''

import os
from os import system, name

from PIL import Image, ImageDraw
import face_recognition


# Helper function.
# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return

# Main implementation function.
# Input: image file
# Return: nothing, 
#   prints detected face count and all faces locations within the image,
#   displays input picture with detectd face bounded in red rectangle.
def face_detect_add_rectangle(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        return # invalid input. we should not proceed.
    imgFile = imgFile.strip()
    if os.path.exists(imgFile) == False:
        print("Image file does not exists. Please check file and try again.")
        return # invalid input. we should not proceed.
    
    image = face_recognition.load_image_file(imgFile)
    faceLocations = face_recognition.face_locations(image)
    numberOfFaces = len(faceLocations)
    print("Found {} face(s) in this picture.".format(numberOfFaces))
    
    pilImage = Image.fromarray(image)
    drawHandle = ImageDraw.Draw(pilImage)

    for top, right, bottom, left in faceLocations:
        print("A face is located at pixel location Top: {}, Left {},Bottom: {}, Right: {}".format(top, left, bottom, right))
        drawHandle.rectangle([left, top, right, bottom], outline="red", width=3)

    pilImage.show()
    return


# Application execution main entry point.
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone")
    print("Module 1: Critical Thinking Assignment")
    print("  Option 1: ...write Python code to detect faces in an image...\n")

    imgFile = "people-h.png"  # local image file
    face_detect_add_rectangle(imgFile)  # call implementation function