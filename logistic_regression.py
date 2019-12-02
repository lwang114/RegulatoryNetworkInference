
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

# X is the sample by gene matrix 
# Y is a vector of labels
#need to define label from prior models
# 0.5  in the training set and .5 in the test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

# Those parameters try to restrain the variables used in the model
#need to identify the correct penalty from previous model

model = LogisticRegression(penalty='l2', C=0.2)  
model.fit(X_train, y_train)

# Apply the model on the test set
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)

#rough draft
