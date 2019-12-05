import numpy as np
import json
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import random
import time
import matplotlib.pyplot as plt
from utils import *
from evaluate import *

EPS = 1e-30
if __name__ == '__main__':
  task = [0, 2]
  #-----------# 
  # Read Data #
  #-----------#  
  exp_dir = 'inferred_lasso/'
  datafile = '../merlin-p_inferred_networks/yeast_networks/expression/NatVar.txt'  
  gold_network_file = '../merlin-p_inferred_networks/yeast_networks/gold/MacIsaac2.NatVar.txt'  
  names, X = read_gene_expression(datafile) 
  X = np.asfortranarray(X.T)
  n_g = X.shape[-1]
  
  #-------------------------------#
  # Model training and prediction #
  #-------------------------------#
  if 0 in task:
    #model = LassoCV()
    alpha = 0.05
    model = Lasso(alpha=alpha)
    
    X_predict = np.nan * np.ones(X.shape)
    edge_weights = []
    r2_scores = []
    avg_r2_score = 0.
    avg_pcc_score = 0.


    # For each gene in the network, run a LASSO regression to determine a subset of potential TFs for the gene;
    # the coefficients of LASSO are used as confidence score for the regulatory relations;
    # the prediction performance is evaluated based on average R2 score and PCC score between the groundtruth 
    # expressions and the true expressions across all genes
    for i in range(n_g):
      begin_time = time.time()
      js_neq_i = [j for j in range(n_g) if j != i]
      model.fit(X[:, js_neq_i], X[:, i])  
      weights_i = model.coef_
      #print(weights_i)
      edge_weights.append([(j, w) for j, w in enumerate(weights_i.tolist()) if w != 0])
      X_predict[:, i] = model.predict(X[:, js_neq_i])
      pcc = pearson_correlation_coefficient(X_predict[:, i], X[:, i]) 
      r2_score = model.score(X[:, js_neq_i], X[:, i])
      
      r2_scores.append('\t'.join([str(i), names[i], str(r2_score)]))
      avg_r2_score += r2_score
      avg_pcc_score += pcc
      print('Takes %.5f s to train on gene %d' % (time.time()-begin_time, i))
      
    avg_r2_score /= n_g
    avg_pcc_score /= n_g
    with open(exp_dir+'edge_weights.json', 'w') as f:
      json.dump(edge_weights, f, indent=4, sort_keys=True)
    
    predict_network = []
    # TODO: try different definition of the edge probabilities
    for i, w_i in enumerate(edge_weights):
      for j, w_ij in w_i:
        prob = 1  
        for k, w_jk in edge_weights[j]:
          if k == i and w_jk >= w_ij:
            prob = w_ij / (w_jk + w_ij)
            break
              
        predict_network.append('\t'.join([names[i], names[j], str(prob)]))  

    with open(exp_dir+'predicted_network.txt', 'w') as f:
      f.write('\n'.join(predict_network))
    
    np.savez(exp_dir+'predicted_expression.npz', X_predict)
    with open(exp_dir+'r2_scores.txt', 'w') as f:
      f.write('\n'.join(r2_scores))
    
    print('average R2 score: %.1f' % avg_r2_score)
    print('average PCC score: %.1f' % avg_pcc_score)
 
  #------------#
  # Evaluation #
  #------------#
  if 1 in task:
    edge_based_metrics(exp_dir, gold_network_file)
  
  #not completed
  #if 3 in task:
  
  #model = LogisticResgression
  #logreg = LogisticRegression()
  #need to define features
  # fit the model with data
  #logreg.fit(X_train,y_train)
#
  #y_pred=logreg.predict(X_test)
