#subset of gene expression from the gold network to train a small subset and evaluate
#saves time
import numpy as np
import json
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import random
import time
import matplotlib.pyplot as plt
from utils import read_gene_expression, pearson_correlation_coefficient
from evaluate import *

# For each gene in the target set, run a LASSO regression to determine a subset of the TF set to be potential TFs for the gene;
# X_tf: the gene expressions for the TF genes;
# tf_names: names of the TFs;
# X_target: the gene expressions for the target genes;
# target_names: names of the targets;
# the coefficients of LASSO are used as confidence score for the regulatory relations;
# the prediction performance is evaluated based on average R2 score and PCC score between the groundtruth 
# expressions and the true expressions across all genes    
def lasso_GRN(X_tf, X_target, tf_names, target_names, alpha=0.01, max_num_edges=30000):
  n_tf = X_tf.shape[1]
  n_trg = X_target.shape[1]
  model = Lasso(alpha=alpha)
    
  X_predict = np.nan * np.ones(X_target.shape)
  edge_weights = []
  r2_scores = []
  avg_r2_score = 0.
  avg_pcc_score = 0.
  # XXX
  #n_trg = 100
  for i in range(n_trg):
    begin_time = time.time()
    js_neq_i = [j for j in range(n_tf) if j != i and pearson_correlation_coefficient(X_target[:, i], X_tf[:, j]) != 1.]
    model.fit(X_tf[:, js_neq_i], X_target[:, i])  
    weights_i = model.coef_
    #print(weights_i)
    edge_weights.append([(j, w) for j, w in enumerate(weights_i.tolist()) if w != 0])
    X_predict[:, i] = model.predict(X_tf[:, js_neq_i])
    pcc = pearson_correlation_coefficient(X_predict[:, i], X_target[:, i]) 
    r2_score = model.score(X_tf[:, js_neq_i], X_target[:, i])
    
    r2_scores.append('\t'.join([str(i), target_names[i], str(r2_score)]))
    avg_r2_score += r2_score
    avg_pcc_score += pcc
    print('Takes %.5f s to train on gene %d' % (time.time()-begin_time, i))
    
  avg_r2_score /= n_trg
  avg_pcc_score /= n_trg
  with open(exp_dir+'edge_weights.json', 'w') as f:
    json.dump(edge_weights, f, indent=4, sort_keys=True)
  
  predict_network = []
  # TODO: try different definition of the edge probabilities; try limiting the maximum number of edges
  for i_trg, w_i in enumerate(edge_weights):
    for i_tf, w_ij in w_i:
      if abs(w_ij) > 0:
        prob = abs(w_ij)
        predict_network.append('\t'.join([tf_names[i_tf], target_names[i_trg], str(prob)]))  

  with open(exp_dir+'predicted_network.txt', 'w') as f:
    f.write('\n'.join(predict_network))
  
  np.savez(exp_dir+'predicted_expression.npz', X_predict)
  with open(exp_dir+'r2_scores.txt', 'w') as f:
    f.write('\n'.join(r2_scores))
  
  print('average R2 score: %0.1f' % avg_r2_score)
  print('average PCC score: %0.1f' % avg_pcc_score)
 
if __name__ == '__main__':
  task = [1, 2, 3]
  #-----------# 
  # Read Data #
  #-----------#
  # XXX
  exp_dir = 'inferred_lasso_given_tf_target/' 
  target_datafile = 'data/target_expressions.npy'
  target_name_file = 'data/targets.txt'
  tf_datafile = 'data/tf_expressions.npy'
  tf_name_file = 'data/regulators.txt'

  gold_network_file = '../merlin-p_inferred_networks/yeast_networks/gold/MacIsaac2.NatVar.txt'  
  X_tf = np.load(tf_datafile) 
  with open(tf_name_file, 'r') as f:
    tf_names = f.read().strip().split('\n') 
  X_tf = np.asfortranarray(X_tf.T)
  
  X_target = np.load(target_datafile) 
  with open(target_name_file, 'r') as f:
    target_names = f.read().strip().split('\n') 
  X_target = np.asfortranarray(X_target.T) 

  #-------------------------------#
  # Model training and prediction #
  #-------------------------------#
  if 0 in task:
    alpha = 0.00001 
    #model = LassoCV()
    lasso_GRN(X_tf, X_target, tf_names, target_names, alpha)
  if 1 in task:
    exp_dir = 'inferred_lasso_given_tf_target_combined/'
    alpha = 0.00001
    target_datafiles = ['data/target_stress_expressions.npy', 'data/target_KO_expressions.npy']
    tf_datafiles = ['data/tf_stress_expressions.npy', 'data/tf_KO_expressions.npy']
    for trg_file, tf_file in zip(target_datafiles, tf_datafiles):
      X_tf_1 = np.asfortranarray(np.load(tf_file).T)
      X_trg_1 = np.asfortranarray(np.load(trg_file).T)
      
      X_tf = np.concatenate([X_tf, X_tf_1])
      X_target = np.concatenate([X_target, X_trg_1])
    
    print('X_tf.shape: ', X_tf.shape)   
    lasso_GRN(X_tf, X_target, tf_names, target_names, alpha)
  
  #------------#
  # Evaluation #
  #------------#
  if 2 in task:    
    edge_based_metrics(exp_dir, gold_network_file)
  if 3 in task:
    gene_based_metrics(exp_dir, gold_network_file, tf_names, target_names) 
