from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import random
import time
import matplotlib.pyplot as plt
import numpy as np

def edge_based_metrics(exp_dir, gold_network_file, debug=False):
  begin_time = time.time()
  pred_network_genes = []
  pred_edge_dict = {}
  f = open(exp_dir+'predicted_network.txt')
  i = 0
  for line in f:
    # XXX
    #if i > 10:
    #  break
    #i += 1
    
    parts = line.split()
    g1, g2, prob = parts[0], parts[1], parts[2]
    pred_edge_dict[g1+'_'+g2] = float(prob)
    if g1 not in pred_network_genes:
      pred_network_genes.append(g1)
    if g2 not in pred_network_genes:
      pred_network_genes.append(g2)
  f.close()
  if debug:
    print('predict network genes: ', pred_network_genes)

  gold_network_genes = []
  gold_edge_dict = {}
  f = open(gold_network_file)    
  i = 0
  for line in f:
    # XXX
    #if i > 10:
    #  break
    #i += 1
    parts = line.split()
    g1, g2 = parts[0], parts[1]
    if g1 not in gold_network_genes:
      gold_network_genes.append(g1)
    if g2 not in gold_network_genes:
      gold_network_genes.append(g2)
    gold_edge_dict[g1+'_'+g2] = 1
  f.close()
  if debug:
    print('gold network genes: ', gold_network_genes)

  # y_probs are the predicted probabilities of the label to be 1    
  y_probs = []
  # y_true are the true labels
  y_true = [] 
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
  if debug:
    print('y_probs: ', y_probs[:10])
    print('y_true: ', y_true[:10])
  print('Take %.2f s to extract network data' % (time.time() - begin_time))
  
  # XXX
  fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
  # Print ROC curve
  plt.figure()
  plt.plot(fpr,tpr)
  plt.title('ROC curve for LASSO')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.show() 
  
  plt.figure()
  prec, recall, thres = precision_recall_curve(y_true, y_probs)  
  if debug:
    print('thresholds: ', thres)
  # Plot PR curve
  plt.plot(recall, prec) 
  plt.title('PR curve for LASSO')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.show()
  plt.close()

  # Print AUC
  auc = np.trapz(tpr, fpr)
  print('AUC:', auc)   
  ap = average_precision_score(y_true, y_probs)
  print('Average precision: ', ap)

# TODO:
def gene_based_metrics(exp_dir, gold_network_file, tf_names, target_names, overlap_thres=0.9, debug=False):
  begin_time = time.time()
  pred_network_genes = []
  pred_edge_dict = {}
  f = open(exp_dir+'predicted_network.txt')
  i = 0
  for line in f:
    # XXX
    #if i > 10:
    #  break
    #i += 1
    
    parts = line.split()
    g1, g2, prob = parts[0], parts[1], parts[2]
    pred_edge_dict[g1+'_'+g2] = float(prob)
    if g1 not in pred_network_genes:
      pred_network_genes.append(g1)
    if g2 not in pred_network_genes:
      pred_network_genes.append(g2)
  f.close()
  if debug:
    print('predict network genes: ', pred_network_genes)

  gold_network_genes = []
  gold_edge_dict = {}
  f = open(gold_network_file)    
  i = 0
  for line in f:
    # XXX
    #if i > 10:
    #  break
    #i += 1
    parts = line.split()
    g1, g2 = parts[0], parts[1]
    if g1 not in gold_network_genes:
      gold_network_genes.append(g1)
    if g2 not in gold_network_genes:
      gold_network_genes.append(g2)
    gold_edge_dict[g1+'_'+g2] = 1
  f.close()
  if debug:
    print('gold network genes: ', gold_network_genes)

  y_true = []
  y_probs = [] 
  y_true_tf = {}
  y_pred_tf = {}
  y_true_trg = {}
  y_pred_trg = {} 
  for i_tf, g_tf in enumerate(tf_names):
    y_true_tf[g_tf] = []
    y_pred_tf[g_tf] = []
    for i_tg, g_tg in enumerate(target_names):
      if g_tg not in y_pred_trg:
        y_pred_trg[g_tg] = []
      if g_tg not in y_true_trg:
        y_true_trg[g_tg] = []

      eKey = g_tf + '_' + g_tg
      if eKey in gold_edge_dict:
        y_true.append(1)
        y_true_tf[g_tf].append(1)
        y_true_trg[g_tg].append(1)
      else:
        y_true.append(0)
        y_true_tf[g_tf].append(0)
        y_true_trg[g_tg].append(0)

      if eKey in pred_edge_dict:
        y_probs.append(pred_edge_dict[eKey])
        y_pred_tf[g_tf].append(pred_edge_dict[eKey])
        y_pred_trg[g_tg].append(pred_edge_dict[eKey])
      else:
        y_probs.append(0)
        y_pred_tf[g_tf].append(0)
        y_pred_trg[g_tg].append(0)
  
  # XXX
  fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
  # Print ROC curve
  plt.figure()
  plt.plot(fpr,tpr)
  plt.title('ROC curve for LASSO')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
    
  prec, recall, _ = precision_recall_curve(y_true, y_probs)  
  # Plot PR curve
  plt.figure()
  plt.plot(recall, prec) 
  plt.title('PR curve for LASSO')
  plt.xlabel('Recall')
  plt.ylabel('Precision')

  # Print AUC
  auc = np.trapz(tpr, fpr)
  print('AUC:', auc)   
  ap = average_precision_score(y_true, y_probs)
  print('Average precision: ', ap)

  plt.show()
  plt.close()

  n_tf_correct = 0.
  for g_tf in tf_names:
    ap = average_precision_score(y_true_tf[g_tf], y_pred_tf[g_tf])
    if ap >= overlap_thres:
      n_tf_correct += 1
  
  print('Number of identifiable TFs: ', n_tf_correct) 

  n_trg_correct = 0.
  for g_trg in target_names:
    ap = average_precision_score(y_true_trg[g_trg], y_pred_trg[g_trg])
    if ap >= overlap_thres:
      n_trg_correct += 1
  
  print('Number of identifiable targets: ', n_trg_correct) 


