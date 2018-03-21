 
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:44:57 2018

@author: coucoulico
"""

from sys import argv, path
path.append ("../ingestion_program") # Contains libraries you will need
from data_manager import DataManager  # such as DataManager
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
import numpy as np
from tempfile import mkdtemp
#importation des differents scalers pour mettre les varriables dans une meme echelle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#importation des outils de selection des variables
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

#importation des module pour se debarasser des varriable manquante et aberrantes
from sklearn.preprocessing import Imputer

#pour la reduction de dimension
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

#des moduls du clustring
from sklearn.cluster import FeatureAgglomeration

class Preprocessor(BaseEstimator):
    def __init__(self):
        self.transformer = PCA(n_components=0.95)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X)
 
    #des methode pour la selection de varriables
    #1-
    def selectFeatures(self, X, y=None):
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        return model
    
    def selectFeatures2(self, X, y=None):
        clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        return model
    
    
    #utilisation des piplines pour la combinaision
    #pour les valeur manquante on a utilisé la class Imputer 
    #pour la normalisation des varriable sous la meme echelle on à preferé le scaler standard 
    #puisque ce la n a aucune influence sur les resultat
    
    #dans ces 3 methode pip0,pip1,pip2 on a cobiné le pca avec les differentes methode de selection de varriable
    def pip0(self, X, y=None):
         estimators = [('imputer',Imputer()),('scaler',StandardScaler()),
                       ('reduct_dim', PCA()),
                       ('clustring',FeatureAgglomeration())]
         cachedir = mkdtemp()
         pipe = Pipeline(estimators)#, memory=cachedir
         pipe.fit(X, y)
         return pipe
    
     
    def pip1(self, X, y=None):
         estimators = [('imputer',Imputer()),('scaler',MinMaxScaler()),
                       ('reduct_dim', PCA()),
                       ('clustring',FeatureAgglomeration())]
         cachedir = mkdtemp()
         pipe = Pipeline(estimators, memory=cachedir)
         #pipe.fit(X, y)
         self.transformer=pipe
         print(self.transformer)
         return pipe
    
    
    #dans ces 3 methode pip3,pip4,pip5 on a cobiné le LLE avec les differentes methode de selection de varriable
    def pip2(self, X, y=None):
         estimators = [('imputer',Imputer()),('scaler',StandardScaler()),
                       ('reduce_dim', LocallyLinearEmbedding()),
                       ('clustring',FeatureAgglomeration())]
         pipe = Pipeline(estimators)
         pipe.fit(X, y)
         return pipe
    
     
    def pip3(self, X, y=None):
         estimators = [('imputer',Imputer()),('scaler',MinMaxScaler()),
                       ('reduce_dim', LocallyLinearEmbedding()),
                       ('clustring',FeatureAgglomeration())]
         cachedir = mkdtemp()
         pipe = Pipeline(estimators, memory=cachedir)
         pipe.fit(X, y)
         return pipe
     

    
        
        
        

if __name__=="__main__":
    # We can use this to run this file as a script and test the Preprocessor
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../sample_data"
        output_dir = "../results" # Create this directory if it does not exist
    else:
        input_dir = argv[1]
        output_dir = argv[2];
    
    basename = 'houseprice'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print D
    
    Prepro = Preprocessor()
    
    
   
    X=np.copy(D.data['X_train'])
    y=np.copy(D.data['Y_train'])
    x_valid=np.copy(D.data['X_valid'])
    x_test=np.copy(D.data['X_valid'])
    #test de la PCA
    
    print("*teste de la methode PCA pour la reduction de dimension avec le Standardscaler pour la normalisation*" )
    
    transformer1=Prepro.pip0(X,y)
    
    D.data['X_train'] = transformer1.fit_transform(X, y)
    D.data['X_valid'] = transformer1.transform(x_valid)
    D.data['X_test'] = transformer1.transform(x_test)
    
    print("*** Transformed data with : PCA ***")
    print(D)
    
    
    
    #test de la LLE

    print("*teste de la methode LLA pour la reduction de dimension avec le Standardscaler pour la normalisation*" )
    transformer2=Prepro.pip2(X,y)
    D.data['X_train'] = transformer2.fit_transform(X, y)
    D.data['X_valid'] = transformer2.transform(x_valid)
    D.data['X_test'] = transformer2.transform(x_test)
    
    print("*** Transformed data : with LLE ***")
    print(D)
    
    
    #reduction de dimension avec la selection des varriables
    #cette prepmier nous le reduit on un espace de 9 dimensions
    model_selection = Prepro.selectFeatures(X, y)
    D.data['X_train'] = model_selection.transform(X)
    D.data['X_valid']=model_selection.transform(x_valid)
    D.data['X_test']=model_selection.transform(x_test)
    estimators = [('imputer',Imputer()),('scaler',MinMaxScaler()),
                       ('clustring',FeatureAgglomeration())]
    cachedir = mkdtemp()
    
    #puis passe au etapes suivantes qu on regroupe dans des piplines
    pipe = Pipeline(estimators)
    D.data['X_train']=pipe.fit_transform(D.data['X_train'],D.data['Y_train'])
    D.data['X_valid']=pipe._transform(D.data['X_valid'])
    D.data['X_test']=pipe._transform(D.data['X_test'])
    
   
   
   
    
    
    print("*** Transformed data :  selction des features avec LinearSVC des svm ***")
    print(D)
    
    
    
    
    
    #cette deuxieme methodes de selction nous permet de reduire la dimension de nos varriables à 8
    model_selection2 = Prepro.selectFeatures2(X, y)
    D.data['X_train'] = model_selection2.transform(X)
    D.data['X_valid']=model_selection2.transform(x_valid)
    D.data['X_test']=model_selection2.transform(x_test)
    #on utilise le meme pipline que celui du test qui precede pour les autres traitements
    pipe = Pipeline(estimators)
    D.data['X_train']=pipe.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid']=pipe._transform(D.data['X_valid'])
    D.data['X_test']=pipe._transform(D.data['X_test'])
    
    print("*** Transformed data : selction des features avec  ExtraTreesClassifier des ensembles  ***")
    print(D)
    
    #si il s'agit des espaces de tres grande dimension on peut combiner les deux methodes features_selection et PCA ou LLE
    #la maniere et la suivante 
    #1-on fait un selection des varriables
    model_selection2 = Prepro.selectFeatures2(X, y)
    D.data['X_train'] = model_selection.transform(X)
    D.data['X_valid']=model_selection.transform(x_valid)
    D.data['X_test']=model_selection.transform(x_test)
    #on utilise le meme pipline que celui du test qui precede pour les autres traitements
    
    #puis on reduit l'espace resultant avec la PCA par exemple
    transformer3=Prepro.pip0(X,y)
    
    D.data['X_train'] = transformer3.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = transformer3.transform(D.data['X_valid'])
    D.data['X_test'] = transformer3.transform(D.data['X_test'])
    print("*** Transformed data combinaison des deux methodes ***")
    print(D)
    
    #remarque :dans notre cas la PCA est suffisante car on n'a que 18 varriables 
    #mais cela devra etre plus efficace si on se met devant un espace de 50 ou 100 varriable explicatives

  
    # Here show something that proves that the preprocessing worked fine
   
