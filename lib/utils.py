import os, sys

from datetime import datetime


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('!!ERROR!! in basics->createDirectory:  Creating directory. ' +  directory)

def string2value(string):
    if string.isdigit():
        return int(string)
    elif all(x in string for x in ['[',']']):
        return eval(string.split()[0])    
    else:
        try:
            float(string)
            return float(string)
        except:
            return string

def isComment(string):
    if string[0] in ['//', '#']:
        return True
    else:
        return False

def isTrue(string):
    if string in ['yes', 'Yes', 'True', 'true']:
        return True
    else:
        return False








