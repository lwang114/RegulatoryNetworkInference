import numpy as np
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

def pearson_correlation_coefficient(x, y):
  return x @ y / max(np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2), EPS)

def find_degrees_of_regulation(network_file, out_prefix=''):
  f = open(network_file, 'r')
  tf2target, target2tf = {}, {}
  for line in f:
    e = line.split()
    if e[0] not in tf2target:
      tf2target[e[0]] = [e[1]]
    if e[1] not in target2tf:
      target2tf[e[1]] = [e[0]]
    tf2target[e[0]].append(e[1])
    target2tf[e[1]].append(e[0])
  f.close()

  with open('tf_to_target.txt', 'w') as f:
    f.write('TF name\t Target names\n')
    for tf in sorted(tf2target): 
      f.write(' '.join([tf] + tf2target[tf]))
      f.write('\n')
  
  n_multiple_tfs = 0  
  with open('target_to_tf.txt', 'w') as f:
    f.write('Target name\t TF names\n')
    for target in sorted(target2tf): 
      f.write(' '.join([target] + target2tf[target]))
      f.write('\n') 
      if len(target2tf[target]) >= 2:
        n_multiple_tfs += 1
  print('Number of target genes with multiple TFs: ', n_multiple_tfs)
  print('Number of target genes with multiple TFs: ', float(n_multiple_tfs)/len(target2tf)) 

'''
def generate_sythesize_data(tf_names, target_names, p, file_prefix=''):
  network = []
  target2tf = [ for ]
  tf2ids = {tf_name: i for i, tf_name in enumerate(tf_names)}
  target2ids = {target_name: i for i, target_name in enumerate(target_names)}

  for tf_name in tf_names:
    for target_name in target_names:
      if random.random() < p:
        network.append(' '.join([tf_name, target_name]))

        target2tf[target2ids[target_name]] = tf_name
        
  # Generate tf expressions

  # Generate target expressions

  with open(file_prefix+'random_network.txt', 'w') as f:
    f.write('\n'.join(network))
'''

def find_overlap_target_regulator(network_file, out_prefix=''):
  f = open(network_file, 'r')
  regulators, targets = [], []
  for line in f:
    e = line.split()
    regulators.append(e[0])
    targets.append(e[1])
  f.close()

  regulators = set(regulators)
  targets = set(targets)
  with open('regulators.txt', 'w') as f:
    f.write('\n'.join(list(regulators)))

  with open('targets.txt', 'w') as f:
    f.write('\n'.join(list(targets)))
  
  n_overlap = len(regulators.intersection(targets))
  n_tot = len(regulators.union(targets))
  print('Number of overlapping genes: ', n_overlap)
  print('Total Number of genes: ', n_tot)
  print('Percent of intersection for TF: ', float(n_overlap) / len(regulators))
  print('Percent of intersection: ', float(n_overlap) / float(n_tot))

def extract_expression_given_set(expression_file, set_file, out_file_prefix=''):
  gene_names, X = read_gene_expression(expression_file)
  with open(set_file, 'r') as f:
    gene_sets = f.read().strip().split('\n')  
  
  X_set = []
  for g in gene_sets:
    for i, g1 in enumerate(gene_names):
      if g == g1:
        X_set.append(X[i])

  np.save(out_file_prefix+'expressions', np.array(X_set))

if __name__ == '__main__':
  tasks = [0, 2]
  datafile = '../merlin-p_inferred_networks/yeast_networks/expression/NatVar.txt'  
  network_file = '../merlin-p_inferred_networks/yeast_networks/gold/MacIsaac2.NatVar.txt'  
  if 0 in tasks:
    find_overlap_target_regulator(network_file)
  if 1 in tasks:
    find_degrees_of_regulation(network_file)
  if 2 in tasks:
    extract_expression_given_set(datafile, 'regulators.txt', out_file_prefix='tf_')
    extract_expression_given_set(datafile, 'targets.txt', out_file_prefix='target_')
