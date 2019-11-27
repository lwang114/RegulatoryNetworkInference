import numpy as np
import json
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import random
import time
import matplotlib.pyplot as plt

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

  print('Average Pearson Coefficient by the random network: ', np.mean(pccs))
  np.savez(file_prefix+'.npz', X_predict)
  with open(file_prefix+'_pcc.json', 'w') as f:
    json.dump(pccs, f, indent=4, sort_keys=True)
  return np.mean(pccs), np.mean(r2_scores)

def pearson_correlation_coefficient(x, y):
  return x @ y / max(np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2), EPS)

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
              
        predict_network.append(' '.join([names[i], names[j], str(prob)]))  

    with open(exp_dir+'predicted_network.txt', 'w') as f:
      f.write('\n'.join(predict_network))
    
    np.savez(exp_dir+'predicted_expression.npz', X_predict)
    with open(exp_dir+'r2_scores.txt', 'w') as f:
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
    f = open(exp_dir+'r2_scores.txt')
    r2_scores = []
    for line in f:
      r2_scores.append(float(line.split()[-1]))
    f.close()
    print('Average R2 score of LASSO: ', np.mean(r2_scores)) 
    # Correlation score
    
    pred_edge_dict = {}
    f = open(exp_dir+'predicted_network.txt')
    for line in f:
      parts = line.split()
      g1, g2, prob = parts[0], parts[1], parts[2]
      pred_edge_dict[g1+'_'+g2] = float(prob)
    f.close()
    # y_probs are the predicted probabilities of the label to be 1    
    y_probs = []
    y_true = [] 
    # y_true are the true labels
   
    gold_network_genes = []
    gold_edge_dict = {}
    f = open(gold_network_file)    
    i = 0
    for line in f:
      # XXX
      if i > 10:
        break
      i += 1
      parts = line.split()
      g1, g2 = parts[0], parts[1]
      gold_network_genes += [g1, g2]
      gold_edge_dict[g1+'_'+g2] = 1
    f.close()

    for g1 in gold_network_genes:
      for g2 in gold_network_genes:
        eKey = g1+'_'+g2
        # Check the probability of the edge in the predicted network
        if eKey in gold_edge_dict:
          y_true.append(1)  
        else:
          y_true.append(0)
            
        if eKey in pred_edge_dict:
          y_probs.append(pred_edge_dict[eKey])
        else:
          y_probs.append(0)
        
    # XXX
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    # Print ROC curve
    plt.plot(fpr,tpr)
    plt.show() 
    plt.title('ROC curve for LASSO')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    prec, recall, _ = precision_recall_curve(y_true, y_probs)  
    # Plot PR curve
    plt.plot(recall, prec) 
    plt.show() 
    plt.title('PR curve for LASSO')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Print AUC
    auc = np.trapz(tpr, fpr)
    print('AUC:', auc)   
    ap = average_precision_score(y_true, y_probs)
    print('Average precision: ', ap)
