from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
import argparse


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', dest='f', help='features filename .csv')		
	return parser.parse_args()

std  = {'mutag': 0, 'ptc': 0, 'nci1': 0, 'nci109': 0, 'enzymes': 0, 'proteins': 0, 'collab': 0, 'imdb_binary': 0, 'imdb_multi': 0, 'reddit_binary': 0,'reddit_multi5k': 0, 'reddit_multi10k': 0}
args = parse_args()

#USE: run training_rf.py -f mutag_features.csv

#----- Loading Features ----------------------------------

filename    =  args.f
standarize  =  std[filename[:-13:]]
df          =  pd.read_csv("features/"+filename)

# --------------------------------------------------------------------------------------

# Preparing data and labels

n_classes = len(df['labels'].unique())

if n_classes == 2:
   df['labels']=df['labels'].replace(-1,0)


X = df.iloc[:,range(0,len(df.columns)-1)].values
y = df['labels'].values

#---------------------------------------------------------
print(filename)
print(standarize)
print(df.columns.values)
print(X.shape)
if 'node_attribs_0' in df.columns.values:
	print("Training With node attributes")
else:
	print("Training Without node attributes")	
	


seed = 666
ext_folds = 10
inner_folds = 10


skf = StratifiedKFold(n_splits=ext_folds, shuffle=True, random_state=seed)

folds_acc = []

for train_index, test_index in skf.split(X, y):


	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	model      =  RandomForestClassifier()
	std_scaler =  StandardScaler()


	n_param    =  [10,100,300,500,700,1000]
		
	if standarize:
		pipe = Pipeline([('std_scaler', std_scaler), ('rf', model)])
	else:	
		pipe = Pipeline([('rf', model)])

	param_grid = dict(rf__n_estimators=n_param)
	
	clf = GridSearchCV(pipe, param_grid=param_grid, cv=inner_folds, scoring='accuracy', n_jobs=-1)
	y_score = clf.fit(X_train, y_train)

	
	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_) 

	
	print("Detailed classification report:")
	print()
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	print()
	
	y_true, y_pred  =  y_test, clf.predict(X_test)   
	acc             =  accuracy_score(y_true, y_pred)
	folds_acc.append(acc)
		
	print("Acc on testing fold: %0.2f" % acc) # accuracy on the unseen test (fold) set for a particula training fold
	print(); 
	
			
folds_acc = np.array(folds_acc)
print("Final accuracy: %0.2f +/- %0.02f" % (folds_acc.mean()*100, folds_acc.std()*100))			

