import numpy as np
import json
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import cross_validate
import random
import time

def read_gene_expression(datafile):
  f = open(datafile, 'r')
  i = 0
  condition_names = None
  gene_names = [] 
  X = []
  for line in f:
    # XXX
    #if i > 10:
    #  break
    if i == 0:
      condition_names = line.strip().split()[1:]
    else:
      parts = line.strip().split()
      gene_names.append(parts[0])
      X.append(list(map(float, parts[1:])))
    i += 1
  f.close()
  print('----- Data Summary -----')
  print('Num. of genes: %d' % len(gene_names))
  print('Num. of conditions: %d' % len(condition_names))

  return gene_names, np.array(X)
  
if __name__ == '__main__':
  task = 0
  #-----------# 
  # Read Data #
  #-----------#  
  datafile = '../merlin-p_inferred_networks/yeast_networks/expression/NatVar.txt'  
  names, X = read_gene_expression(datafile) 
  X = np.asfortranarray(X.T)
  n_g = X.shape[-1]
  
  #-------------------------------#
  # Model training and prediction #
  #-------------------------------#
  if task == 0:
    #model = LassoCV()
    alpha = 0.05
    model = Lasso(alpha=alpha)
    
    X_predict = np.nan * np.ones(X.shape)
    edge_weights = []
    r2_scores = []
    avg_r2_score = 0.
    for i in range(n_g):
      begin_time = time.time()
      js_neq_i = [j for j in range(n_g) if j != i]
      model.fit(X[:, js_neq_i], X[:, i])  
      weights_i = model.coef_
      #print(weights_i)
      edge_weights.append([(j, w) for j, w in enumerate(weights_i.tolist()) if w != 0])
      X_predict[:, i] = model.predict(X[:, js_neq_i])
      r2_score = model.score(X[:, js_neq_i], X[:, i])
      r2_scores.append(' '.join([str(i), str(r2_score), names[i]]))
      avg_r2_score += r2_score
      print('Takes %.5f s to train on gene %d' % (time.time()-begin_time, i))
      
    avg_r2_score /= n_g
    with open('edge_weights.json', 'w') as f:
      json.dump(edge_weights, f, indent=4, sort_keys=True)

    np.savez('predicted_expression.npz', X_predict)
    with open('r2_scores.json', 'w') as f:
      f.write('\n'.join(r2_scores))
    
    print('average R2 score: %.1f' % avg_r2_score)

  #------------#
  # Evaluation #
  #------------#
  if task == 1:
    # R^2 score on validation set
    # Correlation score
    # TODO: AUROC, AUPR
    print   
