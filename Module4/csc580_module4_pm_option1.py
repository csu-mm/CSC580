'''
MS - Artificial Intelligence and Machine Learning
Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone
Module 4: Portfolio Milestone
Professor: Dr. Brian Holbert
Created by Mukul Mondal
June 2, 2026

Problem statement: 
Option #1: Implementing Facial Recognition

For this Portfolio Project Milestone, you will build on the programming requirements presented 
    in Critical Thinking Assignment, Module 1, Option 1.

In this Milestone, you will implement a Python program that uses facial recognition to determine 
    if an individual face is present in a group of faces. For example, your program will use 
    facial recognition to determine that the following individual:

...with recent advancements in deep learning, the accuracy of face recognition has improved. 
In this course, learn how to develop a face recognition system that can detect faces in images, 
identify the faces, and even modify faces with "digital makeup" like you've experienced in 
popular mobile apps. Find out how to set up a development environment. Discover tools you can 
leverage for face recognition. See how a machine learning model can be trained to analyze images
and identify facial landmarks. Learn the steps involved in coding facial feature detection, 
representing a face as a set of measurements, and encoding faces. Additionally, learn how 
to repurpose and adjust pre-existing systems.

Submit your Python code and image data using a zip file entitled:
CSC526_MidTermPortfolio _Option_1_last_name_first_name.zip
'''

import os
from os import system, name

from PIL import Image, ImageDraw
from PIL import ImageFont
import face_recognition


# Helper function.
# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return

# input:
#   faceName: name of the known face
#   imgFile: image file of the known face, having single image
#   txtColor: (optional) color used for writing the name
# output:
#   returns above known face encoding data.
#   also, draws face landmarks, bounding rectangle, writes the name, then displays the image
def draw_Landmarks_GetEncoding_SingleKnownFace(imgFile: str, faceName: str, txtColor: str="black"):
    if imgFile is None or len(imgFile.strip()) < 1:
        return # invalid input. we should not proceed.
    if faceName is None or len(faceName.strip()) < 1:
        return # invalid input. we should not proceed.
    imgFile = imgFile.strip()
    faceName = faceName.strip()

    image = face_recognition.load_image_file(imgFile)
    encoding = face_recognition.face_encodings(image)[0]
    face_location = face_recognition.face_locations(image)
    face_landmarks_list = face_recognition.face_landmarks(image)
    known_image = Image.fromarray(image)
    known_draw = ImageDraw.Draw(known_image)

    for (top, right, bottom, left), landmarks in zip(face_location, face_landmarks_list):        
        known_draw.rectangle(((left, top), (right, bottom)), outline="green", width=2) # Draw bounding box
        font = ImageFont.truetype("arial.ttf", size=20)  #font = ImageFont.load_default()
        known_draw.text((left, top - 10), faceName, fill=txtColor, font=font)

        # Draw facial landmarks
        for _, points in landmarks.items():
            for point in points:
                known_draw.ellipse( (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="red" )
    
    known_image.show()
    return encoding


# identifies 'known face' among faces
# if 'found' writes the name of the 'known face'
#       otherwise, writes 'Unknown' as name.
# also, draws face landmarks, bounding rectangles, writes the names, then display the image with all faces.
# Returns: True - if found, False - otherwise
def faceIdentify(knownImgFile: str, knownFaceName: str, testImgFile: str, txtColor: str="black"):
    if knownFaceName is None or len(knownFaceName.strip()) < 1:
        return # invalid input. we should not proceed.
    if knownImgFile is None or len(knownImgFile.strip()) < 1:
        return # invalid input. we should not proceed.
    if testImgFile is None or len(testImgFile.strip()) < 1:
        return # invalid input. we should not proceed.
    
    knownFaceName = knownFaceName.strip()
    knownImgFile = knownImgFile.strip()
    testImgFile = testImgFile.strip()

    known_face_encoding = draw_Landmarks_GetEncoding_SingleKnownFace(knownImgFile, knownFaceName, txtColor)
    if known_face_encoding is None or len(known_face_encoding) < 1:
        print("Known face processing error. Please check the input file.")
        return

    # Load test face(s) image file
    image = face_recognition.load_image_file(testImgFile)

    # Detect face(s) in the test image file
    face_locations = face_recognition.face_locations(image)
    if face_locations is None or len(face_locations) < 1:
        print("Test face image file processing error. Please check the input file.")
        return
    face_encodings = face_recognition.face_encodings(image, face_locations)
    if face_encodings is None or len(face_encodings) < 1:
        return
    face_landmarks_list = face_recognition.face_landmarks(image)
    if face_landmarks_list is None or len(face_landmarks_list) < 1:
        return
    
    # Convert to PIL image
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    knownFaceFound: bool = False
    # Process each face
    for (top, right, bottom, left), encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list ):
        # Compare test face encoding with known face encoding
        matches = face_recognition.compare_faces([known_face_encoding], encoding)
        if matches[0]:
            knownFaceFound = True
        name = knownFaceName if matches[0] else "Unknown"
        draw.rectangle(((left, top), (right, bottom)), outline="green", width=2)  # Draw bounding box

        # write identified name or "Unknown"
        font = ImageFont.truetype("arial.ttf", size=30)  #font = ImageFont.load_default()
        draw.text((left, top - 25), name, fill=txtColor, font=font)

        # Draw facial landmarks
        for _, points in landmarks.items():
            for point in points:
                draw.ellipse( (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="red" )
    
    pil_image.show()    
    return knownFaceFound



# Application execution main entry point.
# It calls above functions to perform all tasks needed for: "Module 4 Option 1: Implementing Facial Recognition".
# I did test with 2 sets of input image data files. User may try any data files.
# Output:
#   It displays both image files with face landmarks, face bounding rectangle.
#   It also displays 'user choosen face name' on reference face.
#   It displays 'Unknown' on all other non-matching faces.
#   It also displays result in text message with full information.
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone")
    print("Module 4: Portfolio Milestone")
    print("  Option 1: Implementing Facial Recognition.\n")
    
    # imgFile1 = local image file. Please update path for your image file. Known face.
    # imgFile2 = local image file. Please update path for your image file. Unknown face(s).
    # imgFile1 = os.getcwd() + "/datafiles/Module4/1person.png"

    found: bool = False
    # testing # 1 # ok  # these image files are collection from online.
    # to use this test, please comment out test 2.
    imgFile1 = "Simon Helberg.png"
    imgFile2 = "bigbang-group.png"
    found = faceIdentify(imgFile1, "Simon Helberg", imgFile2, "white")

    # testing # 2 # ok # these image data files are copied from this Module 4.
    # to use this test, please comment out test 1.
    #imgFile1 = "shutterstock141032905--250.jpg"  # 1 person , known face
    #imgFile2 = "shutterstock169945061--250.jpg"  # people , find known face among faces in this image.
    #found = faceIdentify(imgFile1, "Mr. Module4", imgFile2, "blue")

    if found:
        print(f"Face from image file: {imgFile1} , found in image file: {imgFile2}")
    else:
        print(f"Face from image file: {imgFile1} , not found in image file: {imgFile2}")

"""
Face encoding:
The process of taking an image of a face and turning it into a set of measumements.
A real face-encoding system will capture a large number of face measurements (typically 128 or more).
Instead of trying to decide on 128 ways to measure a face, we'll use machine learning to create those measurements.
-----

Development environment:
    Window 10 Prof
    VS Code
    Python 3.10.0

Needed installs:
    pip install Pillow
    pip install dlib
    pip install face_recognition
"""
