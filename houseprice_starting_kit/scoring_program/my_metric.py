#Sklearn version of r2_score

from sklearn import metrics

def my_r2_score(solution, prediction):
    return metrics.r2_score(solution, prediction)