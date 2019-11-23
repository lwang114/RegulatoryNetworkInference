import numpy as np
import json
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import cross_validate
import random
import time

EPS = 1e-30
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

def predict_with_random_network(X, p, file_prefix='random_network_predicted_expression'):
  n_c, n_g = X.shape
  model = LinearRegression() 
  pccs = [] 
  r2_scores = []
  X_predict = np.zeros(X.shape)
  for i in range(n_g):
    begin_time = time.time()
    edges_i = []
    while len(edges_i) == 0: 
      edges_i = [j for j in range(n_g) if random.random()<=p and j != i]
    model.fit(X[:, edges_i], X[:, i])
    X_predict[:, i] = model.predict(X[:, edges_i]) 
    r2_score = model.score(X[:, edges_i], X[:, i])
    pccs.append(pearson_correlation_coefficient(X_predict[:, i], X[:, i]))
    r2_scores.append(r2_score)
    print('Take %.5f s to process gene %d' % (time.time() - begin_time, i)) 

  print('Average Pearson Coefficient: ', np.mean(pccs))
  np.savez(file_prefix+'.npz', X_predict)
  with open(file_prefix+'_pcc.json', 'w') as f:
    json.dump(pccs, f, indent=4, sort_keys=True)
  return np.mean(pccs), np.mean(r2_scores)

# TODO
def pearson_correlation_coefficient(x, y):
  return x @ y / max(np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2), EPS)

if __name__ == '__main__':
  task = [0, 1]
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
  if 0 in task:
    #model = LassoCV()
    alpha = 0.05
    model = Lasso(alpha=alpha)
    
    X_predict = np.nan * np.ones(X.shape)
    edge_weights = []
    r2_scores = []
    avg_r2_score = 0.
    avg_pcc_score = 0.
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
      
      r2_scores.append(' '.join([str(i), names[i], str(r2_score)]))
      avg_r2_score += r2_score
      avg_pcc_score += pcc
      print('Takes %.5f s to train on gene %d' % (time.time()-begin_time, i))
      
    avg_r2_score /= n_g
    avg_pcc_score /= n_g
    with open('edge_weights.json', 'w') as f:
      json.dump(edge_weights, f, indent=4, sort_keys=True)

    np.savez('predicted_expression.npz', X_predict)
    with open('r2_scores.txt', 'w') as f:
      f.write('\n'.join(r2_scores))
    
    print('average R2 score: %.1f' % avg_r2_score)
    print('average PCC score: %.1f' % avg_pcc_score)


  #---------------------------#
  # Random network prediction #
  #---------------------------#
  if 1 in task:
    p = 3802. / 114. * 537. / (5661. * 5660. / 2) # TODO: find the right ratio  
    print('connect an edge with probability: %.5f' % p)
    rep = 5
    avg_pcc = 0.
    avg_r2 = 0. 
    for r in range(rep):
      avg_pcc_r, avg_r2_r = predict_with_random_network(X, p, file_prefix='random_%d' % r)
      avg_pcc += avg_pcc_r
      avg_r2 += avg_r2_r

    print('Overall average R2 score for the random network: %.5f' % (avg_r2 / rep)) 
    print('Overall average PCC score for the random network: %.5f' % (avg_pcc / rep))
  
  #------------#
  # Evaluation #
  #------------#
  if 2 in task:
    # R^2 score on validation set
    f = open('r2_scores.txt')
    r2_scores = []
    for line in f:
      r2_scores.append(float(line.split()[1]))
    print(np.mean(r2_scores)) 
    # Correlation score
    # TODO: AUROC, AUPR
    print   
    
  #------------#
  # AUROC CURVE #
  #------------#
  if 3 in task:
# X are the predicted labels    
# Y and Z are the true labels
  Y = open('D:\\wittney2\\CS_598\\gold_standard\\MacIsaac2.NatVar')    
  y_true = Y
  y_probs = X
  fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probs, pos_label=0)

# Print ROC curve
   plt.plot(fpr,tpr)
   plt.show() 

# Print AUC
   auc = np.trapz(tpr,fpr)
   print('AUC:', auc)
