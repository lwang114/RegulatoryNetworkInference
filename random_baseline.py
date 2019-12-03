import numpy as np
from utils import *
from sklearn.linear_model import LinearRegression
from evaluate import *

def predict_with_random_network(X, p, file_prefix='random_network_predicted_expression'):
  n_c, n_g = X.shape
  model = LinearRegression() 
  pccs = [] 
  r2_scores = []
  edge_weights = []
  predict_network = []

  X_predict = np.zeros(X.shape)
  
  # Compute R2 and PCC
  for i in range(n_g):
    begin_time = time.time()
    edges_i = []
    # Randomly choose a set of TFs
    while len(edges_i) == 0: 
      edges_i = [j for j in range(n_g) if random.random()<=p and j != i]
    model.fit(X[:, edges_i], X[:, i])
    weights_i = model.coef_
    #print(weights_i) 
    edge_weights.append([(j, 1) for j, w in enumerate(weights_i.tolist()) if w != 0])
    
    for i, w_i in enumerate(edge_weights):
      for j, w_ij in w_i:
        prob = w_ij 
        predict_network.append(' '.join([names[j], names[i], str(prob)]))  

    X_predict[:, i] = model.predict(X[:, edges_i]) 
    r2_score = model.score(X[:, edges_i], X[:, i])
    pccs.append(pearson_correlation_coefficient(X_predict[:, i], X[:, i]))
    r2_scores.append(r2_score)
    print('Take %.5f s to process gene %d' % (time.time() - begin_time, i)) 
  
  with open(file_prefix+'_predicted_network.txt', 'w') as f:
    f.write('\n'.join(predict_network))
  with open(file_prefix+'_r2_scores.txt', 'w') as f:
    f.write('\n'.join())   
  print('Average Pearson Coefficient by the random network: ', np.mean(pccs))
  print('Average R2 score by the random network: ', np.mean(r2_scores))
  return np.mean(pccs), np.mean(r2_scores)

if __name__ == '__main__':
  tasks = [2]
  #-----------# 
  # Read Data #
  #-----------#  
  exp_dir = 'inferred_random/'
  datafile = '../merlin-p_inferred_networks/yeast_networks/expression/NatVar.txt'  
  gold_network_file = '../merlin-p_inferred_networks/yeast_networks/gold/MacIsaac2.NatVar.txt'  
  names, X = read_gene_expression(datafile) 
  X = np.asfortranarray(X.T)
  n_g = X.shape[-1]
  rep = 1

  #-------------------------------#
  # Model training and prediction #
  #-------------------------------#
  if 1 in tasks:
    p = 3802. / 114. * 537. / (5661. * 5660. / 2) # TODO: find the right ratio  
    
    avg_pcc = 0.
    avg_r2 = 0.
    for r in range(rep):
      avg_pcc_r, avg_r2_r = predict_with_random_network(X, p, file_prefix=exp_dir+'random_%d' % r)
      avg_pcc += avg_pcc_r
      avg_r2 += avg_r2_r
 
    print('Overall average R2 score for the random network: %.5f' % (avg_r2 / rep)) 
    print('Overall average PCC score for the random network: %.5f' % (avg_pcc / rep))
  
  #-----------------------#
  # Edge-based Evaluation #
  #-----------------------#
  if 2 in tasks:
    for r in range(rep):
      edge_based_metrics(exp_dir+'random_%d_' % r, gold_network_file, debug=True) 
