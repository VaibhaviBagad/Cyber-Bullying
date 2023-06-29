# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:31:26 2022

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:19:23 2022

@author: admin
"""

import speech_recognition as sr
import pyttsx3
 
# # Initialize the recognizer
# r = sr.Recognizer()
 
# # Function to convert text to
# # speech
# source2="E:/cyber bullying/audio1.mp3"
# audio2 = r.listen(source2)
             
#             # Using google to recognize audio
# MyText = r.recognize_google(audio2)
# MyText = MyText.lower()
 
# print("Did you say "+MyText)

r = sr.Recognizer()
with sr.AudioFile('audio1.wav') as source:
    audio = r.record(source)

command = r.recognize_google(audio)
print(command)
