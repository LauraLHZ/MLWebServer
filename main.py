import sys
import numpy as np
import time
import itertools
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def ml_linear_regression(X_trn, y_trn, X_tst, y_tst):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    time_1 = time.time()
    reg.fit(X_trn, y_trn)
    time_2 = time.time()
    acc_trn = reg.score(X_trn, y_trn)
    time_3 = time.time()
    acc_tst = reg.score(X_tst, y_tst)
    time_4 = time.time()
    
    trn_cnf_matrix = confusion_matrix(y_trn, reg.predict(X_trn))
    tst_cnf_matrix = confusion_matrix(y_tst, reg.predict(X_tst))
    
    return acc_trn, acc_tst, time_2 - time_1, time_3 - time_2, time_4 - time_3, trn_cnf_matrix, tst_cnf_matrix

def ml_logistic_regression(X_trn, y_trn, X_tst, y_tst):
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    time_1 = time.time()
    reg.fit(X_trn, y_trn)
    time_2 = time.time()
    acc_trn = reg.score(X_trn, y_trn)
    time_3 = time.time()
    acc_tst = reg.score(X_tst, y_tst)
    time_4 = time.time()
    
    trn_cnf_matrix = confusion_matrix(y_trn, reg.predict(X_trn))
    tst_cnf_matrix = confusion_matrix(y_tst, reg.predict(X_tst))
    
    return acc_trn, acc_tst, time_2 - time_1, time_3 - time_2, time_4 - time_3, trn_cnf_matrix, tst_cnf_matrix

def ml_knn(X_trn, y_trn, X_tst, y_tst):
    from sklearn.neighbors import KNeighborsClassifier
    time_1 = time.time()
    knn = KNeighborsClassifier()
    knn.fit(X_trn, y_trn)
    time_2 = time.time()
    acc_trn = knn.score(X_trn, y_trn)
    time_3 = time.time()
    acc_tst = knn.score(X_tst, y_tst)
    time_4 = time.time()
    
    trn_cnf_matrix = confusion_matrix(y_trn, knn.predict(X_trn))
    tst_cnf_matrix = confusion_matrix(y_tst, knn.predict(X_tst))
    
    return acc_trn, acc_tst, time_2 - time_1, time_3 - time_2, time_4 - time_3, trn_cnf_matrix, tst_cnf_matrix

def ml_svm(X_trn, y_trn, X_tst, y_tst):
    from sklearn.svm import SVC
    clf = SVC()
    time_1 = time.time()
    clf.fit(X_trn, y_trn)
    time_2 = time.time()
    acc_trn = clf.score(X_trn, y_trn)
    time_3 = time.time()
    acc_tst = clf.score(X_tst, y_tst)
    time_4 = time.time()
    
    trn_cnf_matrix = confusion_matrix(y_trn, clf.predict(X_trn))
    tst_cnf_matrix = confusion_matrix(y_tst, clf.predict(X_tst))
    
    return acc_trn, acc_tst, time_2 - time_1, time_3 - time_2, time_4 - time_3, trn_cnf_matrix, tst_cnf_matrix

def ml_rf(X_trn, y_trn, X_tst, y_tst):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators= 100)
    time_1 = time.time()
    clf.fit(X_trn, y_trn)
    time_2 = time.time()
    acc_trn = clf.score(X_trn, y_trn)
    time_3 = time.time()
    acc_tst = clf.score(X_tst, y_tst)
    time_4 = time.time()
    
    trn_cnf_matrix = confusion_matrix(y_trn, clf.predict(X_trn))
    tst_cnf_matrix = confusion_matrix(y_tst, clf.predict(X_tst))
    
    return acc_trn, acc_tst, time_2 - time_1, time_3 - time_2, time_4 - time_3, trn_cnf_matrix, tst_cnf_matrix

def plot_confusion_matrix(cm, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print ('No enough arguments.')
    else:
        trn_filename = sys.argv[1]
        tst_filename = sys.argv[2]

        trn_data = np.loadtxt(trn_filename, delimiter=',')
        tst_data = np.loadtxt(tst_filename, delimiter=',')

        X_trn = trn_data[:, :-1]
        y_trn = trn_data[:, -1]
        X_tst = tst_data[:, :-1]
        y_tst = tst_data[:, -1]

        acc_trn = 0.0
        acc_tst = 0.0
        trn_time = 0.0
        training_time = 0.0
        predict_on_trn_time = 0.0
        predict_on_tst_time = 0.0

        ml_method_no = sys.argv[3]
        # if ml_method_no == 'linear_regression':
        #     acc_trn, acc_tst, training_time, predict_on_trn_time, predict_on_tst_time, trn_cnf_matrix, tst_cnf_matrix = ml_linear_regression(X_trn, y_trn, X_tst, y_tst)
        if ml_method_no == 'logistic_regression':
            acc_trn, acc_tst, training_time, predict_on_trn_time, predict_on_tst_time, trn_cnf_matrix, tst_cnf_matrix = ml_logistic_regression(X_trn, y_trn, X_tst, y_tst)
        elif ml_method_no == 'knn':
            acc_trn, acc_tst, training_time, predict_on_trn_time, predict_on_tst_time, trn_cnf_matrix, tst_cnf_matrix = ml_knn(X_trn, y_trn, X_tst, y_tst)
        elif ml_method_no == 'svm':
            acc_trn, acc_tst, training_time, predict_on_trn_time, predict_on_tst_time, trn_cnf_matrix, tst_cnf_matrix = ml_svm(X_trn, y_trn, X_tst, y_tst)
        elif ml_method_no == 'random_forest':
            acc_trn, acc_tst, training_time, predict_on_trn_time, predict_on_tst_time, trn_cnf_matrix, tst_cnf_matrix = ml_rf(X_trn, y_trn, X_tst, y_tst)

        np.set_printoptions(precision=2)

        fig_1 = plt.figure()
        plot_confusion_matrix(trn_cnf_matrix, title= ml_method_no + '\nConfusion matrix on training set')

        fig_2 = plt.figure()
        plot_confusion_matrix(tst_cnf_matrix, title=ml_method_no + '\nConfusion matrix on test set')

        fig_1.savefig('./results/trn_cm.png')
        fig_2.savefig('./results/tst_cm.png')

        #plt.show()

        print ('Accuracy on training set: ' + str(acc_trn))
        print ('Accuracy on test set: ' + str(acc_tst))
        print ('Training time: ' + str(training_time))
        print ('Prediction time on training set: ' + str(predict_on_trn_time))
        print ('Prediction time on test se: ' + str(predict_on_tst_time))

        f = open("./results/accuracy.txt", "w")
        f.write(str(acc_trn) + '\n' + str(acc_tst) + '\n' + str(training_time) + '\n' + str(predict_on_trn_time) + '\n' + str(predict_on_tst_time))
        f.close()
