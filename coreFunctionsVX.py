import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import random
#from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from  sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def createHyperParameters(seedRandom):
    criterion_random = ['gini','entropy'] 
    split_random = ['best','random'] 
    min_samples_split_random= random.randint(1,10)/10.0
    min_samples_leaf_random= random.randint(1,5)/10.0
    min_weight_fraction_leaf_random=random.randint(0,5)/10.0
    min_impurity_decrease_random=random.randint(0,10)/10.0
    depth_random=random.randint(1,5)
    
    
    hyperparams={'criterion':'gini', 'splitter':'best', 'max_depth':None,\
                 'min_samples_split':2, 'min_samples_leaf':1, \
                 'min_weight_fraction_leaf':0.0, 'max_features':None, \
                 'random_state':seedRandom, 'max_leaf_nodes':None,\
                 'min_impurity_decrease':0.0, 'min_impurity_split':None, \
                 'class_weight':None, 'presort':True}
    
    hyperparams['criterion'] = criterion_random[random.randint(0,len(criterion_random)-1)]
    hyperparams['splitter'] = split_random[random.randint(0,len(criterion_random)-1)]
    hyperparams['max_depth'] = 10
    hyperparams['min_samples_split'] = min_samples_split_random
    hyperparams['min_samples_leaf'] = min_samples_leaf_random
    hyperparams['min_weight_fraction_leaf'] = min_weight_fraction_leaf_random
    hyperparams['min_impurity_decrease'] = min_impurity_decrease_random
    hyperparams['min_impurity_split'] = None
    hyperparams['class_weight']: None 
    hyperparams['presort']: True
    hyperparams['random_state']: seedRandom
    return hyperparams

def plot(seed): 
    
    filename='immuno.csv'
    random.seed(seed)
    np.random.seed(seed)
    data = pd.read_csv(filename,sep='\t')
    data=data.sample(frac=1).reset_index(drop=True)
    X=data.values[:,0:7]
    y=data.values[:,-1]
    #indexList=[0,1,2,3,4,5,6,8,23,24,25]
    names=list(data)[0:7]
    #print(X.shape)
    #print(y.shape)
    #X=X[:,np.r_[indexList]]
    
    params=createHyperParameters(seed)
    
    
    kf=KFold(n_splits=5, random_state=1, shuffle=False)
    acc_history=[]
    split=0
        
    reg = None
    reg=DecisionTreeClassifier()
    reg.set_params(**params)
    for (train_indices, val_indices) in kf.split(X, y):
        split=split+1
        xtrain, xval = X[train_indices], X[val_indices]
        ytrain, yval = y[train_indices], y[val_indices]
#        print(xval)
#        ytrain = ytrain.reshape(-1,1)
#        yval = yval.reshape(-1,1)      
        reg.fit(xtrain,ytrain)
        ypred=reg.predict(xval)
#        ypred2=reg.predict(xtrain)
        
        accuracy = accuracy_score(yval,ypred)
#        print(accuracy)
        acc_history.append(accuracy)
        
        
    ACCVALMIN=np.min(acc_history)
    ACCVALMAX=np.max(acc_history)
    ACCVALMEAN=np.mean(acc_history)
    
    
    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    from sklearn.tree import export_graphviz
    import pydotplus
    from sklearn import tree
    import collections
    
    dot_data = StringIO()
    export_graphviz(reg, out_file=dot_data,
                    feature_names=names,
                    filled=True, rounded=True,
                    special_characters=True)
    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
    
    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
            
    predictions = reg.predict_proba(X)
    ROCAREA=roc_auc_score(y, predictions[:,1])
    ypr = reg.predict(X)
    print("Global Set Accuracy Score : %.4f"%(accuracy_score(y,ypr)))
    print ("Area under ROC Curve : %.4f" %(ROCAREA))
    
    fpr, tpr, _ = roc_curve(y, predictions[:,1])
    plt.clf()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.show()
    
    return(ACCVALMIN,ACCVALMAX,ACCVALMEAN,ROCAREA,params,graph)
    
    