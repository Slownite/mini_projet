'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn import datasets, linear_model

class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''

        self.num_train_samples = len(X)
        if X.ndim>1: self.num_feat = len(X[0])
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = len(y)
        if y.ndim>1: self.num_labels = len(y[0])
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")

        ###### Baseline models ######
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        # Comment and uncomment right lines in the following to choose the model
        self.clf = GaussianNB()
        #self.clf = LinearRegression()
        #self.clf = DecisionTreeRegressor()
        #self.clf = RandomForestRegressor()
        #self.clf = KNeighborsRegressor()

        self.clf.fit(X, y)
        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = len(X)
        if X.ndim>1: num_feat = len(X[0])
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        return self.clf.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self


######## Main function ########
if __name__ == "__main__":
    # Find the files containing corresponding data
    # To find these files successfully:
    # you should execute this "model.py" script in the folder "sample_code_submission"
    # and the folder "public_data" should be in the SAME folder as the starting kit
    path_to_training_data = "../../public_data/houseprice_train.data"
    path_to_training_label = "../../public_data/houseprice_train.solution"
    path_to_testing_data = "../../public_data/houseprice_test.data"
    path_to_validation_data = "../../public_data/houseprice_valid.data"

    # Find the program computing R sqaured score
    path_to_metric = "../scoring_program/my_metric.py"
    import imp
    r2_score = imp.load_source('metric', path_to_metric).my_r2_score

    # use numpy to load data
    X_train = np.loadtxt(path_to_training_data)
    y_train = np.loadtxt(path_to_training_label)
    X_test = np.loadtxt(path_to_testing_data)
    X_valid = np.loadtxt(path_to_validation_data)


    # TRAINING ERROR
    # generate an instance of our model (clf for classifier)
    clf = model()
    # train the model
    clf.fit(X_train, y_train)
    # to compute training error, first make predictions on training set
    y_hat_train = clf.predict(X_train)
    # then compare our prediction with true labels using the metric
    training_error = r2_score(y_train, y_hat_train)


    # CROSS-VALIDATION ERROR
    from sklearn.model_selection import KFold
    from numpy import zeros, mean
    # 3-fold cross-validation
    n = 3
    kf = KFold(n_splits=n)
    kf.get_n_splits(X_train)
    i=0
    scores = zeros(n)
    for train_index, test_index in kf.split(X_train):
        Xtr, Xva = X_train[train_index], X_train[test_index]
        Ytr, Yva = y_train[train_index], y_train[test_index]
        M = model()
        M.fit(Xtr, Ytr)
        Yhat = M.predict(Xva)
        scores[i] = r2_score(Yva, Yhat)
        print ('Fold', i+1, 'example metric = ', scores[i])
        i=i+1
    cross_validation_error = mean(scores)

    # Print results
    print("\nThe scores are: ")
    print("Training: ", training_error)
    print ('Cross-Validation: ', cross_validation_error)

    print("""
To compute these errors (scores) for other models, uncomment and comment the right lines in the "Baseline models" section of the class "model".
To obtain a validation score, you should make a code submission with this model.py script on CodaLab.""")
