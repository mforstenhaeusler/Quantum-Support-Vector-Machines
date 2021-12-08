import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

def normalize(X):
    """ Normalizes Data """
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    
    return (X - x_min)/(x_max - x_min)

def accuracy(y, y_pred, verbose, mode):
    y_true = y[y == y_pred]
    if verbose: 
        print(f'Accuracy on {mode} set: {len(y_true)/len(y) * 100} %')
    else:
        return len(y_true)/len(y)



def compare_models(datasets, models, data_titles, titles, opacity=1, decision_func=False, scikit=False):
    """
    
    Params:
    -------
    datasets : list
               list of datasets
    
    models : list
             list of models
    
    Return:
    -------
    plot of classifiers on each dataset
    """
    start = time()
    i = 1

    for idx, dataset in enumerate(datasets):
        figure = plt.figure(figsize=(27, 20))
        ax = plt.subplot(len(datasets), len(models)+1, i)
        if idx == len(datasets)-1:
            X_train, y_train, X_test, y_test, adhoc_total = dataset
            #X_train, y_train, X_test, y_test, adhoc_total = normalize(X_train), normalize(y_train), normalize(X_test), normalize(y_test), normalize(adhoc_total)
        else:
            X_train, X_test, y_train, y_test = dataset

        xx = np.linspace(X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1, 50) # plotting x range
        yy = np.linspace(X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1, 50) # plotting x range
        X1, X2 = np.meshgrid(yy, yy)
        XX = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        X, y = X_train, y_train
        
        # Plot data    
        ax.scatter(X_train[np.where(y_train[:] == -1),0], X_train[np.where(y_train[:] == -1),1], 
            c='b', marker='o', alpha=0.6, edgecolor='k', label='Class {-1} Train')
        ax.scatter(X_train[np.where(y_train[:] == 1),0], X_train[np.where(y_train[:] == 1),1], 
            c='r', marker='o', alpha=0.6, edgecolor='k', label='Class {1} Train')
        ax.scatter(X_test[np.where(y_test[:] == -1),0], X_test[np.where(y_test[:] == -1),1], 
            c='b', marker='s', alpha=0.6, edgecolor='k', label='Class {-1} Train')
        ax.scatter(X_test[np.where(y_test[:] == 1),0], X_test[np.where(y_test[:] == 1),1], 
            c='r', marker='s', alpha=0.6, edgecolor='k', label='Class {1} Train')
        ax.set_title(data_titles[idx], fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        
        i += 1
        
        for clf, title in zip(models, titles):
            ax = plt.subplot(len(datasets), len(models)+1, i)
            #print(X_train, y_train)
            clf.fit(X_train, y_train.astype(np.float))
            
            if scikit: 
                Z = clf.decision_function(XX).reshape(X1.shape)
            else:
                Z = clf.project(XX).reshape(X1.shape)
            ax.contourf(X1, X2, Z, cmap=plt.cm.RdBu_r, alpha=0.8)
            # decision boundary
            if decision_func: ax.contour(X1, X2, Z, colors="k", levels=[-1, 0, 1], alpha=0.7, linestyles=["--", "-", "--"])
            # support vectors
            if scikit:
                if title=='Quantum':
                    pass
                else:
                    ax.scatter(
                        clf.support_vectors_[:, 0],
                        clf.support_vectors_[:, 1],
                        s=100,
                        linewidth=1,
                        facecolors="none",
                        edgecolors="k",
                        label="Support Vectors"
                    )
            else:
                ax.scatter(
                    clf.sv_X[:,0],
                    clf.sv_X[:,1],
                    s=150,
                    linewidth=1,
                    facecolors="none",
                    edgecolors="k",
                    label="Support Vectors"
                )
            ax.scatter(X[np.where(y[:] == -1),0], X[np.where(y[:] == -1),1], 
                    c='b', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {-1}')
            ax.scatter(X[np.where(y[:] == 1),0], X[np.where(y[:] == 1),1], 
                        c='r', marker='o', alpha=opacity, edgecolor='k', label='Train Data Class {1}')
            # Plot Data
            ax.scatter(X_train[np.where(y_train[:] == -1),0], X_train[np.where(y_train[:] == -1),1], 
                c='b', marker='o', alpha=0.6, edgecolor='k', label='Class {-1} Train')
            ax.scatter(X_train[np.where(y_train[:] == 1),0], X_train[np.where(y_train[:] == 1),1], 
                c='r', marker='o', alpha=0.6, edgecolor='k', label='Class {1} Train')
            ax.scatter(X_test[np.where(y_test[:] == -1),0], X_test[np.where(y_test[:] == -1),1], 
                c='b', marker='s', alpha=0.6, edgecolor='k', label='Class {-1} Train')
            ax.scatter(X_test[np.where(y_test[:] == 1),0], X_test[np.where(y_test[:] == 1),1], 
                c='r', marker='s', alpha=0.6, edgecolor='k', label='Class {1} Train')
            if idx == 0: ax.set_title(title, fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])

            i += 1

        plt.xticks([]) 
        plt.yticks([])
        plt.show()

    end = time()
    print('Time to compute:', (end-start)/60, 'min')

def compare_model_performance(datasets, models, data_titles, titles, scikit=False, verbose=False):
    """
    
    Params:
    -------
    datasets : list
               list of datasets
    
    models : list
             list of models
    
    Return:
    -------
    sol_dict : dict
               dictionary of model accuracy scores 
    """
    start = time()
    i = 1
    sol_dict = {}
    #with tqdm(enumerate(datasets), unit="batch") as comput_range:
    for idx, dataset in enumerate(datasets):
        if idx == len(datasets)-1:
            X_train, y_train, X_test, y_test, adhoc_total = dataset
            #X_train, y_train, X_test, y_test, adhoc_total = normalize(X_train), normalize(y_train), normalize(X_test), normalize(y_test), normalize(adhoc_total)
        else:
            X_train, X_test, y_train, y_test = dataset
        
        sol_dict[data_titles[idx]] = {}
        
        for clf, title in zip(models, titles):
            if verbose: print(f'compute {title} ...')
            clf.fit(X_train, y_train)
            
            if scikit: 
                score = clf.score(X_test, y_test)
            else:
                score = clf.score(X_test, y_test)

            sol_dict[data_titles[idx]][title] = score
            if verbose: print(f'{title} computed!')
    if verbose: print(f'Performance on {data_titles[idx]} computed')
    if verbose: print('\n')

    end = time()
    print('Time to compute:', (end-start)/60, 'min')
    return sol_dict
