 
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
    def __init__(self, t=None):
        #pour ce constructeur il prend un transformer en parametre
        self.transformer=t
        


    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X)
    
    def transform(self, X):
        return self.transformer.transform(X)
     
    """ pour ce qui suit on a creer des methode dans laquelles on cree de Objet transformer oubien des combinaison de ses 
    dernier en utilisant des piplines puis on l'affecte au transformer de notre class prepro"""
    
    #des methode pour la selection de varriables
    #1-une methode pour la selection des variables en utilisant LinearSVC
    def selectFeatures(self, X, y=None):
       
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        self.transformer=model
    
    def selectFeatures2(self, X, y=None):
        clf = ExtraTreesClassifier()
        clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        self.transformer=model
    
    
    #utilisation des piplines pour la combinaision
    #pour les valeur manquante on a utilisé la class Imputer 
    #pour la normalisation des varriable sous la meme echelle on à preferé le scaler standard 
   
    def pip0(self, dim=18):
         """dans cette methode on a utilise le standard scaler pour la normalisation des resultats
         et la PCA pour la reduction de dimension de sorte que la dimension de l'espce reduit est donnée en parametre 
         dim ,par defaut pas de reduction"""
         estimators = [('scalling',StandardScaler()),
                       ('reduct_dim', PCA(n_components =dim ))]
         pipe = Pipeline(estimators)
         self.transformer=pipe
    
    
    
    #dans cette  methode on utilise la LLA pour la reduction de dimension
    def pip1(self, dim=18):
         """ pour cette methode elle a le meme principe que la precedente sauf qu'on a utiliser la methode LLA 
         pour la reduction de la dimension aussi pour la dimension de l'esapace reduit elle est donnée en parametre"""
         estimators = [('imputer',Imputer()),('scaler',StandardScaler()),
                       ('reduce_dim', LocallyLinearEmbedding(n_components=dim))]
         pipe = Pipeline(estimators)
         self.transformer=pipe
    
    
        
        
        

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
    
    print("\n*teste de la methode PCA pour la reduction de dimension avec le Standardscaler pour la normalisation*\n" )
    
    Prepro.pip0(10)
    D.data['X_train'] = Prepro.fit_transform(X, y)
    D.data['X_valid'] = Prepro.transform(x_valid)
    D.data['X_test'] = Prepro.transform(x_test)
    
    print("*** Transformed data with : PCA ***\n")
    print(D)
    
    
    
    #test de la LLE

    print("*\nteste de la methode LLA pour la reduction de dimension avec le Standardscaler pour la normalisation*" )
    Prepro.pip1(5)
    D.data['X_train'] = Prepro.fit_transform(X, y)
    D.data['X_valid'] = Prepro.transform(x_valid)
    D.data['X_test'] = Prepro.transform(x_test)
    
    print("*** Transformed data : with LLE ***\n")
    print(D)
    
    
    #reduction de dimension avec la selection des varriables
    #cette prepmier nous le reduit on un espace de 9 dimensions
    Prepro.selectFeatures(X, y)
    D.data['X_train'] = Prepro.transform(X)
    D.data['X_valid']=Prepro.transform(x_valid)
    D.data['X_test']=Prepro.transform(x_test)
    estimators = [('imputer',Imputer()),('scaler',MinMaxScaler())]
    
    #puis passe au etapes suivantes qu on regroupe dans des piplines
    pipe = Pipeline(estimators)
    D.data['X_train']=pipe.fit_transform(D.data['X_train'],D.data['Y_train'])
    D.data['X_valid']=pipe._transform(D.data['X_valid'])
    D.data['X_test']=pipe._transform(D.data['X_test'])
    
   
   
   
    
    
    print("\n*** Transformed data :  selction des features avec LinearSVC des svm ***\n")
    print(D)
    
    
    
    
    
    #cette deuxieme methodes de selction nous permet de reduire la dimension de nos varriables à 10
    Prepro.selectFeatures2(X, y)
    D.data['X_train'] = Prepro.transform(X)
    D.data['X_valid']=Prepro.transform(x_valid)
    D.data['X_test']=Prepro.transform(x_test)
    #on utilise le meme pipline que celui du test qui precede pour les autres traitements
    pipe = Pipeline(estimators)
    D.data['X_train']=pipe.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid']=pipe._transform(D.data['X_valid'])
    D.data['X_test']=pipe._transform(D.data['X_test'])
    
    print("\n*** Transformed data : selction des features avec  ExtraTreesClassifier des ensembles  ***\n")
    print(D)
    
    
    #si il s'agit des espaces de tres grande dimension on peut combiner les deux methodes features_selection et PCA ou LLE
    #la maniere et la suivante 
    #1-on fait un selection des varriables
    Prepro.selectFeatures2(X, y)
    D.data['X_train'] = Prepro.transform(X)
    D.data['X_valid']=Prepro.transform(x_valid)
    D.data['X_test']=Prepro.transform(x_test)
    
    print('****notre dataFrame apres la selection des variables :***********')
    print(D)
    #on utilise le meme pipline que celui du test qui precede pour les autres traitements
    
    #puis on reduit l'espace resultant avec la PCA par exemple
    transformer3=Prepro.pip0(5)
    
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
    print("\n*** Transformed data combinaison des deux methodes ***\n")
    print(D)
    
    #remarque :dans notre cas la PCA est suffisante car on n'a que 18 varriables 
    #mais cela devra etre plus efficace si on se met devant un espace de 50 ou 100 varriable explicatives

  
    # Here show something that proves that the preprocessing worked fine"""
   
