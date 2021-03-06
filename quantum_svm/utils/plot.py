import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
from quantum_svm.svm.nonlinear_classifier import kernelSVC
from qiskit_machine_learning.algorithms import QSVC
from .utils import normalize

def plot_data(X, y, X_test, y_test, cmap, opacity=0.5):
    """ Plots the data generated from the data_generators. 

    Params:
    ------
    X : nd.array
        Training data

    y : nd.array
        Training labels

    X_test : nd.array
             Testing data

    y_test : nd.array
             Testing labels

    cmap : matplotlib.colormap
           Colormap used for plotting the 2 classes
    
    opacity : int
              Opacity of points in plot 
    """
    fig, ax = plt.subplots(1, figsize=(7,5))
    im = ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, alpha=opacity, label='Train Data')
    im1 = ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=cmap, marker='s', edgecolors='k', alpha=opacity, label='Test Data')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()
    cb = plt.colorbar(im, ax=ax)
    #ax.legend()
    loc = np.arange(-1,1,1)
    cb.set_ticks(loc)
    cb.set_ticklabels(['-1','1'])
    cb.set_label('Classes')
    plt.show()

def plot_SVM(X, y, params=None, baseline_clf=None, opacity=0.6, titles=None, sv=True, hyperplane=True):
    """ Plots the data vs the custom linear classifier vs the baseline model.

    Params:
    -------
    X : nd.array
        Data

    y : nd.array
        Labels 
    
    params : dict
             Dictionary of parameters required to plot the decision function of the custom implemented SVC

    baseline_clf : baseline classifier (usually scikit-learn svc class)
                   Benchmark SVC class 
    
    opacity : int
              Opacity of points in plot  

    titles : list
             List of titles for plot

    sv : bool
         If true, plots support vectors
    
    hyperplane : bool
                 If true, plots decision hyperplane
    """
    xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 50) # plotting x range
    yy = np.linspace(X[:, 1].min(), X[:, 1].max(), 50) # plotting x range

    fig, ax = plt.subplots(1,3, figsize=((23,5)))
    im1 = ax[0].scatter(X[np.where(y[:] == -1),0], X[np.where(y[:] == -1),1], 
                    c='b', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {-1}')
    ax[0].scatter(X[np.where(y[:] == 1),0], X[np.where(y[:] == 1),1], 
                    c='r', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {1}')
    ax[1].scatter(X[np.where(y[:] == -1),0], X[np.where(y[:] == -1),1], 
                    c='b', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {-1}')
    ax[1].scatter(X[np.where(y[:] == 1),0], X[np.where(y[:] == 1),1], 
                    c='r', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {1}')
    ax[2].scatter(X[np.where(y[:] == -1),0], X[np.where(y[:] == -1),1], 
                    c='b', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {-1}')
    ax[2].scatter(X[np.where(y[:] == 1),0], X[np.where(y[:] == 1),1], 
                    c='r', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {1}')
    
    # custom SVC
    if params is not None:
            slope = params['slope']
            intercept = params['intercept']

    if hyperplane:
        ax[1].plot(xx, slope*xx + intercept, 'k-', alpha=0.7, label='Decision Boundary')
        ax[1].plot(xx, xx * slope + intercept - 1/params['weights'][1], 'k--', alpha=0.7)
        ax[1].plot(xx, xx * slope + intercept + 1/params['weights'][1], 'k--', alpha=0.7)
    
    if sv:
        ax[1].scatter(params['sv_X'][:,0], params['sv_X'][:,1], s=150, linewidth=1, facecolors="none", edgecolor='black', label="Support Vectors")
    
    
    # baseline
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = baseline_clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    cs = ax[2].contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.7, linestyles=["--", "-", "--"])
    fmt = {}
    strs = ['first', 'Decision Boundary', 'second',]
    for l, s in zip(cs.levels, strs):
        fmt[l] = s
    ax[2].clabel(cs, cs.levels[1:2], inline=True, fmt=fmt, fontsize=10)
    # plot support vectors
    ax[2].scatter(
        baseline_clf.support_vectors_[:, 0],
        baseline_clf.support_vectors_[:, 1],
        s=150,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
        label="Support Vectors"
    )
    
    
    for i in range(3):
        ax[i].set_xlabel('$x_1$',fontsize=12)
        ax[i].set_ylabel('$x_2$',fontsize=12)
        if titles is not None:
            ax[i].set_title(titles[i], fontsize=16)

    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=13, bbox_to_anchor=(0.5,0))
    plt.show()

def plot_kernel_SVC(X, y, clf_list, cmap, titles=None, kernel=None, opacity=1):
        """ Plots the data vs custom implementation of the kernel SVC vs the baseline model.

        Params:
        -------
        X : nd.array
            Data

        y : nd.array
            Labels 

        clf_list : list
                   List containing all classifiers that are supposed to be plotted    

        titles : list, 
                 List of titles for plot
        
        kernel : str
                 If kernel=='quantum', changes meshgrid density   
                
        opacity : int
                  Opacity of points in plot
        """
        fig, ax = plt.subplots(1,len(clf_list)+1, figsize=((23,5)))
        ax[0].scatter(X[np.where(y[:] == -1),0], X[np.where(y[:] == -1),1], 
                        c='b', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {-1}')
        ax[0].scatter(X[np.where(y[:] == 1),0], X[np.where(y[:] == 1),1], 
                        c='r', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {1}')
        
        if kernel == 'quantum':
            xx = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 20) # plotting x range
            yy = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 20) # plotting x range
            X1, X2 = np.meshgrid(xx, yy)
        else:
            xx = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 50) # plotting x range
            yy = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 50) # plotting x range
            X1, X2 = np.meshgrid(yy, yy)

        XX = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        # clf's
        for idx, clf in enumerate(clf_list):
            if isinstance(SVC(), type(clf)) or isinstance(QSVC(), type(clf)):
                #XX, YY = np.meshgrid(xx, yy)
                #xy = np.vstack([XX.ravel(), YY.ravel()]).T
                Z = clf.decision_function(XX).reshape(X1.shape)

                # plot decision boundary and margins
                ax[idx+1].contourf(X1, X2, Z, cmap=cmap, alpha=0.8)
                cs = ax[idx+1].contour(X1, X2, Z, colors="k", levels=[-1, 0, 1], alpha=0.7, linestyles=["--", "-", "--"])
                fmt = {}
                strs = ['first', 'Decision Boundary', 'second',]
                for l, s in zip(cs.levels, strs):
                    fmt[l] = s
                ax[idx+1].clabel(cs, cs.levels[1:2], inline=True, fmt=fmt, fontsize=10)
                # plot support vectors
                if kernel != 'quantum':
                    ax[idx+1].scatter(
                        clf.support_vectors_[:, 0],
                        clf.support_vectors_[:, 1],
                        s=100,
                        linewidth=1,
                        facecolors="none",
                        edgecolors="k",
                        label="Support Vectors"
                    )
                ax[idx+1].scatter(X[np.where(y[:] == -1),0], X[np.where(y[:] == -1),1], 
                        c='b', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {-1}')
                ax[idx+1].scatter(X[np.where(y[:] == 1),0], X[np.where(y[:] == 1),1], 
                        c='r', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {1}')
            else:
                Z = clf.project(XX).reshape(X1.shape)
                ax[idx+1].contourf(X1, X2, Z, cmap=cmap, alpha=0.8)
                # decision boundary
                cs = ax[idx+1].contour(X1, X2, Z, colors="k", levels=[-1, 0, 1], alpha=0.7, linestyles=["--", "-", "--"])
                fmt = {}
                strs = ['first', 'Decision Boundary', 'second',]
                for l, s in zip(cs.levels, strs):
                    fmt[l] = s
                ax[idx+1].clabel(cs, cs.levels[1:2], inline=True, fmt=fmt, fontsize=10)
                # support vectors
                ax[idx+1].scatter(
                    clf.sv_X[:,0],
                    clf.sv_X[:,1],
                    s=150,
                    linewidth=1,
                    facecolors="none",
                    edgecolors="k",
                    label="Support Vectors"
                )
                ax[idx+1].scatter(X[np.where(y[:] == -1),0], X[np.where(y[:] == -1),1], 
                        c='b', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {-1}')
                ax[idx+1].scatter(X[np.where(y[:] == 1),0], X[np.where(y[:] == 1),1], 
                            c='r', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {1}')

        for i in range(3):
            ax[i].set_xlabel('$x_1$')
            ax[i].set_ylabel('$x_2$')
            ax[i].set_xlim(xx.min(), xx.max())
            ax[i].set_ylim(yy.min(), yy.max())
            if titles:
                ax[i].set_title(titles[i], fontsize=16)
        #ax.axis("tight")
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=13, bbox_to_anchor=(0.5,0))
        plt.show()

def plot_confusion_matrix(y, y_pred_custom, y_pred_baseline, classes, titles=None):
    """ Plots the confusion matrix of the custom implementation vs the baseline model. 
    
    Params:
    -------
    y : nd.array
        True Labels
    
    y_pred_custom : nd.array
                    Predicted labels by custom model 
    
    y_pred_baseline : nd.array
                      Predicted labels by baseline model 
    
    classes : list
              List of colors
    
    titles : list
             List of titles for plot
    """
    cm1 = confusion_matrix(y, y_pred_custom)
    cm2 = confusion_matrix(y, y_pred_baseline)

    fig, ax = plt.subplots(1,2, figsize=((20,6)))
    sns.heatmap(cm1, xticklabels=classes, yticklabels=classes, annot=True, fmt='0.2g', cmap=plt.cm.Blues, ax=ax[0])
    sns.heatmap(cm2, xticklabels=classes, yticklabels=classes, annot=True, fmt='0.2g', cmap=plt.cm.Blues, ax=ax[1])
    bottom, top = ax[0].get_ylim()
    ax[0].set_ylim(bottom + 0.5, top - 0.5)
    ax[1].set_ylim(bottom + 0.5, top - 0.5)
    for i in range(2):
        ax[i].set_xlabel('Predictions')
        ax[i].set_ylabel('Test Set')
        if titles:
            ax[i].set_title(titles[i], fontsize=16)
    
    plt.show()

def plot_adhoc(adhoc_total, X_train, X_test, y_train, y_test):
    """ Adopted from https://lab.quantum-computing.ibm.com/ --> /qiskit-tutorials/qiskit-machine-learning/03_quantum_kernel.ipynb"""
    plt.figure(figsize=(5, 5))
    plt.ylim(0, 2 * np.pi)
    plt.xlim(0, 2 * np.pi)
    plt.imshow(np.asmatrix(adhoc_total).T, interpolation='nearest',
            origin='lower', cmap='RdBu', extent=[0, 2 * np.pi, 0, 2 * np.pi])

    plt.scatter(X_train[np.where(y_train[:] == -1), 0], X_train[np.where(y_train[:] == -1), 1],
                marker='s', facecolors='w', edgecolors='b', label="A train")
    plt.scatter(X_train[np.where(y_train[:] == 1), 0], X_train[np.where(y_train[:] == 1), 1],
                marker='o', facecolors='w', edgecolors='r', label="B train")
    plt.scatter(X_test[np.where(y_test[:] == -1), 0], X_test[np.where(y_test[:] == -1), 1],
                marker='s', facecolors='b', edgecolors='w', label="A test")
    plt.scatter(X_test[np.where(y_test[:] == 1), 0], X_test[np.where(y_test[:] == 1), 1],
                marker='o', facecolors='r', edgecolors='w', label="B test")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title("Ad hoc dataset for classification")

    plt.show()

def plot_datasets(datasets, titles):
    """ Plots a list of dataset.
    
    Params:
    -------
    datasets : list
               List of Datasets
    """
    fig, ax = plt.subplots(2, 3, figsize=(20,10))

    for idx, dataset in enumerate(datasets):
        if idx == 5:
            X_train, y_train, X_test, y_test, adhoc_total = dataset
            #X_train, y_train, X_test, y_test, adhoc_total = normalize(X_train), normalize(y_train), normalize(X_test), normalize(y_test), normalize(adhoc_total)
            i = idx-3 
            ax[1, i].scatter(X_train[np.where(y_train[:] == -1), 0], X_train[np.where(y_train[:] == -1), 1],
                        marker='o', facecolors='b', edgecolors='k', label="class {-1} train")
            ax[1, i].scatter(X_train[np.where(y_train[:] == 1), 0], X_train[np.where(y_train[:] == 1), 1],
                        marker='o', facecolors='r', edgecolors='k', label="class {1} train")
            ax[1, i].scatter(X_test[np.where(y_test[:] == 0), 0], X_test[np.where(y_test[:] == 0), 1],
                        marker='s', facecolors='b', edgecolors='k', label="class {-1} test")
            ax[1, i].scatter(X_test[np.where(y_test[:] == 1), 0], X_test[np.where(y_test[:] == 1), 1],
                        marker='s', facecolors='r', edgecolors='k', label="class {1} test")
            ax[1, i].set_title(titles[idx], fontsize=13)
            ax[1, i].set_xticks([])
            ax[1, i].set_yticks([])
        else:
            X_train, X_test, y_train, y_test = dataset
        
            if idx < 3:
                ax[0, idx].scatter(X_train[np.where(y_train[:] == -1),0], X_train[np.where(y_train[:] == -1),1], 
                    c='b', marker='o', alpha=0.6, edgecolor='k', label='Class {-1} Train')
                ax[0, idx].scatter(X_train[np.where(y_train[:] == 1),0], X_train[np.where(y_train[:] == 1),1], 
                    c='r', marker='o', alpha=0.6, edgecolor='k', label='Class {1} Train')
                ax[0, idx].scatter(X_test[np.where(y_test[:] == -1),0], X_test[np.where(y_test[:] == -1),1], 
                    c='b', marker='s', alpha=0.6, edgecolor='k', label='Class {-1} Train')
                ax[0, idx].scatter(X_test[np.where(y_test[:] == 1),0], X_test[np.where(y_test[:] == 1),1], 
                    c='r', marker='s', alpha=0.6, edgecolor='k', label='Class {1} Train')
                ax[0, idx].set_title(titles[idx], fontsize=13)
                ax[0, idx].set_xticks([])
                ax[0, idx].set_yticks([])
            else:
                i = idx-3    
                ax[1, i].scatter(X_train[np.where(y_train[:] == -1),0], X_train[np.where(y_train[:] == -1),1], 
                    c='b', marker='o', alpha=0.6, edgecolor='k', label='Class {-1} Train')
                ax[1, i].scatter(X_train[np.where(y_train[:] == 1),0], X_train[np.where(y_train[:] == 1),1], 
                    c='r', marker='o', alpha=0.6, edgecolor='k', label='Class {1} Train')
                ax[1, i].scatter(X_test[np.where(y_test[:] == -1),0], X_test[np.where(y_test[:] == -1),1], 
                    c='b', marker='s', alpha=0.6, edgecolor='k', label='Class {-1} Train')
                ax[1, i].scatter(X_test[np.where(y_test[:] == 1),0], X_test[np.where(y_test[:] == 1),1], 
                    c='r', marker='s', alpha=0.6, edgecolor='k', label='Class {1} Train')
                ax[1, i].set_title(titles[idx], fontsize=13)
                ax[1, i].set_xticks([])
                ax[1, i].set_yticks([])
            
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=13)
    plt.show()


