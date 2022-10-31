import sys, os
import re
# for turning dataframe.info into string
import io   

from tabulate import tabulate

sys.path.insert(0, ".")
import utils
import CaseHandler as ch

#from stl import mesh
from trimesh import load
from trimesh import sample
from trimesh import exchange
from trimesh import transformations
from trimesh import creation
from trimesh import repair

from mpl_toolkits import mplot3d
from matplotlib import pyplot

import random

import numpy as np
import pandas as pd


'''

CLASS DATAHANDLER
Read in .stl files, do operations and save data in feature and label matrices

'''

class DataHandler():

    def __init__(self):

        self.CASE               = ch.CaseHandler()

        self.df_features        = pd.DataFrame([], columns=self.CASE.subfeature_names)
        self.df_labels          = pd.read_csv(self.CASE.label_key_filepath, sep='\t', index_col=[0])
        
        self.labels             = self.df_labels.to_numpy()
        self.vertices           = np.zeros((self.CASE.number_of_objects, 
                                            self.CASE.number_of_features, 
                                            3))
        self.features           = np.zeros((self.CASE.number_of_objects, 
                                            self.CASE.number_of_features, 
                                            self.CASE.number_of_subfeatures))
        
        self.stl_filenames      = self._getStlFileNames()



        

    def loadSTLData(self):

        ### get order or transformations, i.e. translation, rotation
        ## order is random
        translation_order   = self._getTransformationOrders('translation')
        rotation_order      = self._getTransformationOrders('rotation')
        
        print(self.stl_filenames)
        for filename_idx, filename in enumerate(self.stl_filenames):
            print(filename)
            filepath            = self.CASE.stl_path + filename
            ### load mesh object using trimesh
            self.mesh           = load(filepath)
            ### transform mesh
            transformation_list = [filename_idx in sublist for sublist in [translation_order, rotation_order]]
            self._transformMesh(transformation_list, self.labels[filename_idx][2], filename_idx)
            ### sample vertices according to given number of features
            vertices            = self.mesh.sample(self.CASE.number_of_features)
            ### save transformed mesh if specified
            if utils.isTrue(self.CASE.data_dict['objectTransformation']['saveTransformedSTL']):
                filepath    = self.CASE.stl_path[:-1] + '_transformed/'+ filename
                with open(filepath, 'w') as fout:
                    fout.write(exchange.stl.export_stl_ascii(self.mesh))
                    fout.close()
                
            self.vertices[filename_idx,:,0] = vertices[:,0]
            self.vertices[filename_idx,:,1] = vertices[:,1]
            self.vertices[filename_idx,:,2] = vertices[:,2]

        x = self.vertices[:,0]
        y = self.vertices[:,1] 
        z = self.vertices[:,2]

        for obj_idx in range(0, self.CASE.number_of_objects):
            for subfeature_idx, subfeature in enumerate(self.CASE.subfeature_names):
                math_pattern = str(self.CASE.data_dict['featureExtraction']['subFeatures'][subfeature])
                math_pattern = math_pattern.replace('x', 'self.vertices[%i,:,0]' %(obj_idx))
                math_pattern = math_pattern.replace('y', 'self.vertices[%i,:,1]' %(obj_idx))
                math_pattern = math_pattern.replace('z', 'self.vertices[%i,:,2]' %(obj_idx))
                math_pattern = math_pattern.replace('n_x', 'self.faces[%i,:,0]' %(obj_idx))
                math_pattern = math_pattern.replace('n_y', 'self.faces[%i,:,1]' %(obj_idx))
                math_pattern = math_pattern.replace('n_z', 'self.faces[%i,:,2]' %(obj_idx))
                #command = 'self.features[%i,:,%i+1] = %s' %(obj_idx, feature_idx, math_pattern)
                command = 'self.features[%i,:,%i] = %s' %(obj_idx, subfeature_idx, math_pattern)
                exec(command)

        del vertices

        ### save data to dataframe
        ## transform dataframe:
        #                  col_1 col_2 ... col_n
        # objIdx ptIdx
        ## df_features
        arrays          = [[i for i in range(self.CASE.number_of_objects) for _ in range(self.CASE.number_of_features)], \
                           list(range(0,self.CASE.number_of_features))*self.CASE.number_of_objects]
        tuples          = list(zip(*arrays))
        index           = pd.MultiIndex.from_tuples(tuples, names=['objIdx', 'pntIdx'])
        self.features   = np.reshape(self.features,(self.CASE.number_of_objects*self.CASE.number_of_features, len(self.CASE.subfeature_names)))
        self.df_features= pd.DataFrame(self.features, index=index, columns=self.CASE.subfeature_names)

        self.df_labels  = pd.DataFrame(self.labels, columns=self.CASE.label_names)
        self.df_labels.index.name = 'objIdx'

    def _getTransformationOrders(self, type):
        order_list = []
        if utils.isTrue(self.CASE.data_dict['objectTransformation'][type]):
            number_of_transformations = int(self.CASE.number_of_objects*self.CASE.data_dict['objectTransformation'][type+'Parameters']['percentage'])
            if self.CASE.transformation_order == 'random':
                order_list = random.sample(range(0, self.CASE.number_of_objects), number_of_transformations)
            elif self.CASE.transformation_order == 'sorted':
                order_list = range(0, number_of_transformations)
            else:
                string = 'ERROR in stlHandler.py: Unknown order mode in %sParameters in stlDict.' %(type)
                print(string)
        return order_list 
        
    def _transformMesh(self, transformation_order, reference, obj_idx):
        ### return transformation matrix for
        ## T -> translation
        ## R -> rotation
        T = transformations.identity_matrix()
        R = transformations.identity_matrix()

        ### translation
        if transformation_order[0]:
            max_value   = self.CASE.stl_dict['objectTransformation']['translationParameters']['maxTranslation']*reference
            x           = np.random.uniform(0, max_value)
            y           = np.random.uniform(0, max_value)
            z           = np.random.uniform(0, max_value)
            vector      = [x, y, z]
            T           = transformations.translation_matrix(vector)
        ### rotation
        if transformation_order[1]:
            max_value   = self.CASE.stl_dict['objectTransformation']['rotationParameters']['maxRotation']
            alpha       = np.random.uniform(0, max_value)
            beta        = np.random.uniform(0, max_value)
            gamma       = np.random.uniform(0, max_value)

        M = transformations.concatenate_matrices(T, R)
        self.mesh.vertices = transformations.transform_points(self.mesh.vertices, M)
        #repair.fix_normals(self.mesh, multibody=False)
        
    def writeDataframesToFile(self):

        self.df_features.to_csv(self.CASE.df_features_path, sep='\t', mode='w')
        self.df_labels.to_csv(self.CASE.df_labels_path, sep='\t', mode='w')

    def visualizePointCloud(self):
        file_idx = 0
        directory = self.CASE.stl_path

        for filename in os.listdir(directory):
            if file_idx < self.CASE.visual_dict['pointCloud']['n']:
                stl_file = directory + filename
                # Create a new plot
                figure = pyplot.figure()
                axes = mplot3d.Axes3D(figure)

                # Load the STL files and add the vectors to the plot
                your_mesh   = mesh.Mesh.from_file(stl_file)
                vertices    = np.around(np.unique(your_mesh.vectors.reshape([int(your_mesh.vectors.size/3), 3]), axis=0), 2)
                axes.scatter(vertices[:,0], vertices[:,1], vertices[:,2])

                # Save plot
                pyplot.savefig(self.CASE.visual_dict['pointCloud']['directory'] + 'Pointcloud_' + filename.replace('.stl',''))
                file_idx += 1


    def visualizeData(self):

        ### plot label data
        for labelIdx0, label0 in enumerate(self.CASE.label_names):
            ax      = self.df_labels.plot(y=label0)
            ax.set_xlabel(self.df_labels.index.name)
            ax.set_ylabel(label0)
            fig     = ax.get_figure()
            path    =   self.CASE.visual_dict['labelData']['directory'] + label0 + '_over_' + self.df_labels.index.name +'.png'
            fig.savefig(path)

            for labelIdx1, label1 in enumerate(self.CASE.label_names):
                if labelIdx1 != labelIdx0:
                    ax      = self.df_labels.plot(x=label1,y=label0)
                    ax.set_xlabel(label1)
                    ax.set_ylabel(label0)
                    fig     = ax.get_figure()
                    path    = self.CASE.visual_dict['labelData']['directory'] + label0 + '_over_' + label1 +'.png'
                    fig.savefig(path)

            ### plot feature data: max value
            df_tmp = self.df_features.groupby(level=0).apply(max)
            df_tmp[label0]= self.df_labels[label0]
            ax      = df_tmp.plot()
            ax.set_xlabel("max(feature)")
            ax.set_ylabel(label0)
            fig     = ax.get_figure()
            path    = self.CASE.visual_dict['labelData']['directory'] + label0 + '_over_max(feature)' +'.png'
            fig.savefig(path)

        ### TODO:
        ## number of points over objIdx


    def createLogOutput(self):
        log_filename = 'log.extractData'
        buf = io.StringIO()
        self.df_features.info(buf=buf)
        string_features_info = buf.getvalue()
        buf = io.StringIO()
        self.df_labels.info(buf=buf)
        string_labels_info = buf.getvalue()

        log_output = self.CASE.returnLogOutputData(log_filename,
                                                   [str(self.df_features),
                                                    string_features_info,
                                                    str(self.df_labels),
                                                    string_labels_info])

        if utils.isTrue(self.CASE.data_dict['printOutputToLogfile']):
            with open(log_filename, 'w') as f:
                f.write(log_output)
            f.close()
        if utils.isTrue(self.CASE.data_dict['printOutputToTerminal']):
            print(log_output)


    def _getStlFileNames(self):

        stl_files   = []
        indices     = []
        directory   = self.CASE.stl_path
        for filename in os.listdir(directory):
            stl_files.append(filename)
            index   = re.findall("[-+]?\d*\.\d+|\d+", filename)[0]
            indices.append(int(index))

        indices     = np.argsort(indices)
        # make sure the correct number of stls are read
        indices     = indices[0:self.CASE.number_of_objects]
        stl_files   = [stl_files[i] for i in indices]


        return stl_files

    def clearMemory(self):
        del self.vertices
        del self.features, self.df_features
        del self.labels, self.df_labels
