# -*- coding: utf-8 -*-
"""
Created on Sat May 30 22:54:01 2020

@author: subham
"""
#importing library
import winsound

def sound(times):
    #winsound.Beep(Frequency, Duration in ms)
    [winsound.Beep(2500,500) for x in range(times)]
