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
    
    def histogramme_val (self, data, valeur):
        sns.distplot(data[valeur])
    
    def correlation (data):
    	corrmat = data.corr()
    	for i in range(40):
    	    corrmat = corrmat.sort_values(by = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price'], ascending = False)
    	    corrmat = corrmat.sort_values(by = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price'], axis = 1, ascending = False)
   		f, ax = plt.subplots(figsize=(12, 9))
    	sns.heatmap(corrmat, vmax=.8, square=True);
        
    def scatter_plot (self, data, cols):
        sns.set()
        sns.pairplot(data[cols], size = 2.5,  palette='afmhot')
        plt.show();
        
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