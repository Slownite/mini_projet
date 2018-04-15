# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

#%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

class visualisation:
    
    def __init__ (self):
        return
    
    """
		Fonction pour afficher un histogramme d'une caracteristique en particulier. Les arguments sont data (XY dans ce cas) et la valeur à analyser représentée par un string (par exemple 'price')
    """
    def histogramme_val (self, data, valeur):
        sns.distplot(data[valeur])
    
	"""
		Fonction qui affiche une heatmap ordonnée de toutes les valeurs de la data_frame afin de voir les influences des caracteristiques les unes sur les autres. L'argument est la data_frame à analyser. (XY ici)
	"""
    def corelation (self, data):
    	corrmat = data.corr()
    	for i in range(40):
    	    corrmat = corrmat.sort_values(by = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price'], ascending = False)
    	    corrmat = corrmat.sort_values(by = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price'], axis = 1, ascending = False)
   	f, ax = plt.subplots(figsize=(12, 9))
    	sns.heatmap(corrmat, vmax=.8, square=True)
        
    """
    	Fonction permettant d'afficher un scatter plot des caracteristiques voulues. Les arguments sont la data_frame (XY ici) et une liste de string des caracteristiques voulues (par exemple ['price', 'sqft_living', 'sqft_basement', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']). Se servir de la heatmap pour selectionner celles qui sont pertinentes.
    """
    def scatter_plot (self, data, cols):
        sns.set()
        sns.pairplot(data[cols], size = 2.5,  palette='afmhot')
        plt.show();
        
    """
    	Fonction qui affiche un graphe d'une caracteristique en fonction d'une autre. Les deux premiers arguments seront l'absisse et l'ordonnée sous forme de strings. Le troisième est le type de graphe souhaité sous forme de string. Les types possibles sont 'scatter', 'box', 'regplot' et 'strip'.
    """
    def graph (self, x, y, tp, data):
        if (tp == 'scatter'):  
            data = pd.concat([data[y], data[x]], axis=1)
            data.plot.scatter(x=x, y=y)
        elif (tp == 'box'):
            f, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x=x, y=y, data=data)
        elif (tp == 'strip'):
            sns.stripplot(x=x, y=y ,data=data)
        elif (tp == 'regplot'):
            sns.regplot(x=x, y=y ,data=data)
