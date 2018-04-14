# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 18:38:44 2018

@author: anino
"""
import re
import matplotlib.pyplot as plt
"""
Ouverture du premier fichier, découpage en liste de string puis traduction en liste de floats
"""
fichier1 = open(r"output_attendu_training.txt")
contenu1 = re.sub(r"(e\+0[0-9])","",fichier1.read())
contenu1 = contenu1.split("\n")
c1 = []
del contenu1[len(contenu1)-1]
for i in contenu1:
    c1.append(float(i))
fichier1.close()

"""
Idem pour le 2eme fichier
"""
fichier2 = open(r"output_training.txt")
contenu2 = re.sub(r"(e\+0[0-9])","",fichier2.read())
contenu2 = contenu2.split("\n")
c2 = []
del contenu2[len(contenu2)-1]
for i in contenu2:
    c2.append(float(i))
fichier2.close()

"""
Ici on défini les points du graphes, avec abscisse et ordonnée selon ce qui
est attendu et ce qui est obtenu, sachant que si c'est identique on est sur la droite y=x rêvée
"""
for i in range (0, len(c2)):
    plt.scatter(c1[i], c2[i])
    plt.scatter(i,i)
"""
Normalement là on affiche le graphe
"""

plt.show()