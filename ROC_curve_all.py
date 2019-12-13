import matplotlib.pyplot as plt
plt.figure()


#ROC curve for all methods
#list of models
#"Baseline", "GENIE3", "inferelator", "LARSEN", "MERLIN", "MERLIN + prior","PGG", "PGG + prior", "TIGRESS", "LASSO + prior"
#define models based on files

gold_standard = 'merlin-p_inferred_networks/yeast_networks/gold/YEASTRACT_Count3.NatVar'


model1 = '../merlin-p_inferred_networks/yeast_networks/inferred_networks/GENIE3.NatVar.txt'  
model2 = '../merlin-p_inferred_networks/yeast_networks/inferred_networks/Inferelator.NatVar.txt' 
model3 = '../merlin-p_inferred_networks/yeast_networks/inferred_networks/LARSEN.NatVar.txt' 
model4 = '../merlin-p_inferred_networks/yeast_networks/inferred_networks/MERLIN.NatVar.txt' 
model5 = '../merlin-p_inferred_networks/yeast_networks/inferred_networks/MERLIN_motif.NatVar.txt' 
model6 = '../merlin-p_inferred_networks/yeast_networks/inferred_networks/PGG.NatVar.txt' 
model7 = '../merlin-p_inferred_networks/yeast_networks/inferred_networks/PGG.NatVar.txt' 
model8 = '../merlin-p_inferred_networks/yeast_networks/inferred_networks/TIGRESS.NatVar.txt'
#baseline model
model9 =  '../predicted_network_Nat_Var.txt'
#need include model with prior information
model10 = '../predicted_network_Nat_Var_prior.txt'
#put models in a list
models = [c(model1, model2,model3, model4, model5, model6, model7, model8, model9, model10)]

for m in models:
    model = m['model'] # select the model
    model.fit(x_train, y_train) 
    y_pred=model.predict(x_test) # predict the test data
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(x_test))

    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right", label = c('GENIE3','inferelator', 'LARSEN', 'MERLIN', 'MERLIN + prior','PGG', 'PGG + prior', 'TIGRESS', 'LASSO + prior', 
                                       'baseline', 'LASSO + prior')
plt.show()
