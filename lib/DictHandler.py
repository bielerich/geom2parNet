import re
import sys
sys.path.insert(0, ".")
import utils
import json 
from datetime import datetime

'''

CLASS DICTHANDLER
Read dict text file from directory pipeline/ and save entries as python dictionary

'''
class DictHandler:

    def __init__(self, directory='pipeline/'):

        self.dict_directory = directory
        if self.dict_directory[-1] != '/':
            self.dict_directory = self.dict_directory + '/'

        self.dict = {}


    def readStlDict(self):

        self.dict_name = 'stlDict'

        dict_path = self.dict_directory + self.dict_name
        self._fillDict(dict_path)


    def readDataDict(self):

        self.dict_name = 'dataDict'
        dict_path = self.dict_directory + self.dict_name
        self._fillDict(dict_path)


    def readLearningDict(self):

        self.dict_name = 'trainingDict'
        dict_path = self.dict_directory + self.dict_name
        self._fillDict(dict_path)


    def readVisualDict(self):

        self.dict_name = 'visualDict'
        dict_path = self.dict_directory + self.dict_name
        self._fillDict(dict_path)


    def _fillDict(self, dict_path):

        with open(dict_path, 'r') as fin:
            string = fin.read()
            fin.close()

        string = re.sub('//(.*?)\n', '\n', string)
        string = re.sub('/\*[\S\n ]+\*/', '\n', string)
        string = re.sub('#(.*?)\n', '\n', string)
        string = re.sub('\n\s*\n', '\n', string)
        string = string.replace('\n', ' ')
        string = string.replace(';', ' ;')

        string = ' ' + string
        string_list = string.split(' ')
        
        clean_list = []

        for item in string_list:
            clean_item = item.replace('{', '')
            clean_item = clean_item.replace('}', '')
            clean_item = clean_item.replace(';', '')
            if all(i not in ["'", '"'] for i in clean_item) and clean_item != '' and clean_item not in clean_list:
                clean_list.append(clean_item)


        for item in clean_list:
            regex = "[\s{};](%s)[\s{}]" %(item)
            replace = '"%s"' %(item)
            string = re.sub(regex, replace, string)

        string = string.replace('{', ' : { ')
        string = string.replace('\n', '')
        string = string.replace(';', ',')
        string = string.replace("'", '"')
        string = re.sub(r'"\s+"', '" : "', string)
        # example: turn "Dense(4,activation="relu")" into "Dense(4,activation='relu')"
        match = re.search(r'=(".*?")', string)
        if match:
            oldMatch = match.group(1)
            newMatch = match.group(1).replace('"', "'")
            string = re.sub(oldMatch, newMatch, string)
        string = re.sub(r'"\s+"', '" : "', string)
        string = re.sub(r',\s+}', '}', string)
        string = re.sub(r'\s', '', string)
        string = string.replace('},}', '}}')
        string = '{' + string + '}'
        string = string.replace('},}', '}}')

        self.dict = json.loads(string) 
        self.dict = self._convertDictValues(self.dict)
        self.dict = self._createDirectories(self.dict)


    def _convertDictValues(self, dictionary):

        for key, val in dictionary.items():
            if isinstance(val, dict):
                dictionary[key] = self._convertDictValues(val)
            else:
                val_converted = utils.string2value(val)
                dictionary[key] = val_converted

        return dictionary

    def _createDirectories(self, dictionary):

        for key, val in dictionary.items():
            if isinstance(val, dict):
                dictionary[key] = self._createDirectories(val)
            else:
                if key == 'directory':
                    if dictionary[key][-1] != '/':
                        dictionary[key] = dictionary[key] + '/'
                    utils.createDirectory(dictionary[key])

        return dictionary

