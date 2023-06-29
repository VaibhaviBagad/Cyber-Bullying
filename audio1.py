# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:16:09 2022

@author: admin
"""

from gtts import gTTS
  
# This module is imported so that we can 
# play the converted audio
import os
  
# The text that you want to convert to audio
mytext = 'not that its any better, but the actual joke youre thinking of is the pair of gay men rape Ryuuji, joke. The okamas just a chill bar lady'
  
# Language in which you want to convert
language = 'en'
  
# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)
  
# Saving the converted audio in a mp3 file named
# welcome 
myobj.save("audio1.wav")
  
# Playing the converted file