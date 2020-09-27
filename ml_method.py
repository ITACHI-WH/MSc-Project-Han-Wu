import scipy.io as sio
import numpy as np
import pdb
import csv
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

DICT_SVM_PARAM = {
	'kernel':['poly', 'rbf'],\
	'C': [ 1e-3, 1e-1, 10],\
	'G': [ 1e-3, 1e-1, 10]
}

DICT_RF_PARAM = {
	'n_estimators': [10, 50, 100, 200, 1000],\
	'max_depth': [10, 20, 40, 60, 80, 100, 150, 200, 500, 1000],\
}

DICT_ANN_PARAM = {
	'layer': [2, 4, 6, 8],\
	'alpha': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],\
	#'alpha': [1e-5, 1, 100]\
}

DICT_KNN_PARAM = {
	'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8,9, 10, 20, 50, 100],\
	#'alpha': [1e-5, 1, 100]\
}


'''
DICT_RF_PARAM = {
	'n_estimators': [10,1000],\
	'max_depth': [10, 100],\
}

DICT_ANN_PARAM = {
	'layer': [2, 8],\
	'alpha': [1e-8,  1],\
	#'alpha': [1e-5, 1, 100]\
}

DICT_KNN_PARAM = {
	'n_neighbors': [1, 10],\
	#'alpha': [1e-5, 1, 100]\
}

DICT_SVM_PARAM = {
	'kernel':['poly', 'rbf'],\
	'C': [1e-5,  1],\
	'G': [1e-5, 1e-1]
}
'''

def print_output(y_test, y_pred, method, score):
	with open('./output/train/ret.txt', 'a') as f:
		f.write("\n\n------------------------------------------------------\n")
		f.write("method = {}, accurate = {}, score = {}".format(method, np.sum(y_pred == y_test)/y_test.shape[0], score))
		labels = ['libx264', 'hevc', 'vp8', 'vp9']
		for l in labels:
			l_test = y_test[y_test == l]
			l_pred = y_pred[y_test == l]
			if l_test.shape[0] != 0:
				f.write("\nLabel = {}, class predict = {:10d}, {:10d}, {:10d}, {:10d}, accurate = {}".format(l, \
					np.sum(l_pred == labels[0]), np.sum(l_pred == labels[1]), np.sum(l_pred == labels[2]), \
					np.sum(l_pred == labels[3]),   np.sum(l_pred == l_test)/l_test.shape[0]))

		f.write(classification_report(y_test,y_pred))



def ml_methods(x_train, x_test, y_train, y_test, coff, method):
	if 'svm' == method:
		model = SVC()
		best_score, best_model, bestk, bestc, bestg = 0, model, 0, 0, 0
		score_test = np.zeros((len(DICT_SVM_PARAM['kernel']), len(DICT_SVM_PARAM['C']), len(DICT_SVM_PARAM['G'])))
		score_train = np.zeros((len(DICT_SVM_PARAM['kernel']), len(DICT_SVM_PARAM['C']), len(DICT_SVM_PARAM['G'])))
		fig, axs = plt.subplots(2,1, figsize = (10, 5))
		for k in range(len(DICT_SVM_PARAM['kernel'])):
			for c in range(len(DICT_SVM_PARAM['C'])):
				for g in range(len(DICT_SVM_PARAM['G'])):
					param_grid = {'kernel': [DICT_SVM_PARAM['kernel'][k]], 'C': [DICT_SVM_PARAM['C'][c]], 'gamma': [DICT_SVM_PARAM['G'][g]]}
					grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=0, cv=5, return_train_score = True)
					grid_search.fit(x_train, y_train)
					best_parameters = grid_search.best_estimator_.get_params()
					model = SVC(kernel = best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'])
					model.fit(x_train, y_train)
					print("best_parameters = {}".format(best_parameters))
					#pdb.set_trace()
					score_test[k, c, g] = model.score(x_test, y_test)
					score_train[k,c, g] = grid_search.best_score_
					if score_train[k,c, g] > best_score:
						best_score = score_train[k,c, g]
						best_model = model
						bestk, bestc, bestg = k, c, g
				axs[0].semilogx(DICT_SVM_PARAM['G'],score_train[k, c, :], '.--', label = 'kernel = {}, C = {}'.format(DICT_SVM_PARAM['kernel'][k], DICT_SVM_PARAM['C'][c]), linewidth=1)


				axs[1].semilogx(DICT_SVM_PARAM['G'],score_test[k, c, :], '.--', label = 'kernel = {}, C = {}'.format(DICT_SVM_PARAM['kernel'][k], DICT_SVM_PARAM['C'][c]), linewidth=1)


		axs[0].plot(DICT_SVM_PARAM['G'][bestg],score_train[bestk, bestc, bestg], 'r*', label = 'best score point', markersize=10)
		axs[0].set_ylabel('training score')
		axs[0].set_xlabel('gamma')
		axs[0].legend(fontsize=6)
		axs[0].grid()
		axs[1].set_ylabel('test score')
		axs[1].set_xlabel('gamma')
		axs[1].legend(fontsize=6)
		axs[1].grid()
		plt.tight_layout()
		plt.savefig('./output/train/svm.png', dpi = 300)
		plt.close('all')
		clf = best_model
		
	elif 'adaboost' == method:
		clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=800, learning_rate=1.0)

	elif 'randomforest' == method:
		model = RandomForestClassifier()
		best_score, best_model, bestn, bestm = 0, model, 0, 0
		score_test = np.zeros((len(DICT_RF_PARAM['n_estimators']), len(DICT_RF_PARAM['max_depth'])))
		score_train = np.zeros((len(DICT_RF_PARAM['n_estimators']), len(DICT_RF_PARAM['max_depth'])))
		fig, axs = plt.subplots(2,1, figsize = (10, 5))
		for n in range(len(DICT_RF_PARAM['n_estimators'])):
			for m in range(len(DICT_RF_PARAM['max_depth'])):
				param_grid = {'n_estimators': [DICT_RF_PARAM['n_estimators'][n]], 'max_depth': [DICT_RF_PARAM['max_depth'][m]]}
				grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=0, cv=5, return_train_score = True)
				grid_search.fit(x_train, y_train)
				best_parameters = grid_search.best_estimator_.get_params()
				model = RandomForestClassifier(n_estimators = best_parameters['n_estimators'], max_depth = best_parameters['max_depth'])
				model.fit(x_train, y_train)
					#pdb.set_trace()
				score_test[n, m] = model.score(x_test, y_test)
				score_train[n, m] = grid_search.best_score_
				print("param = {}, score = {}".format(best_parameters, grid_search.best_score_))
				if score_train[n, m] > best_score:
					best_score = score_train[n, m]
					best_model = model
					bestn, bestm = n, m
			axs[0].plot(DICT_RF_PARAM['max_depth'],score_train[n, :], '.--', label = 'n_estimators = {}'.format(DICT_RF_PARAM['n_estimators'][n]), linewidth=1)
			axs[1].plot(DICT_RF_PARAM['max_depth'],score_test[n, :], '.--', label = 'n_estimators = {}'.format(DICT_RF_PARAM['n_estimators'][n]), linewidth=1)
		axs[0].plot(DICT_RF_PARAM['max_depth'][bestm],score_train[bestn, bestm], 'r*', label = 'best score point', markersize=10)
		axs[0].set_ylabel('training score')
		axs[0].set_xlabel('max_depth')
		axs[0].legend(fontsize=6)
		axs[0].grid()
		axs[1].set_ylabel('test score')
		axs[1].set_xlabel('max_depth')
		axs[1].legend(fontsize=6)
		axs[1].grid()
		plt.tight_layout()
		plt.savefig('./output/train/randomforest.png', dpi = 300)
		plt.close('all')
		clf = best_model
		
	elif 'knn' == method:

		model = neighbors.KNeighborsClassifier()
		best_score, best_model, bestn = 0, model, 0
		score_test = np.zeros((len(DICT_KNN_PARAM['n_neighbors'])))
		score_train = np.zeros((len(DICT_KNN_PARAM['n_neighbors'])))
		fig, axs = plt.subplots(2,1, figsize = (10, 5))
		for n in range(len(DICT_KNN_PARAM['n_neighbors'])):
			param_grid = {'n_neighbors': [DICT_KNN_PARAM['n_neighbors'][n]]}
			grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=0, cv=5, return_train_score = True)
			grid_search.fit(x_train, y_train)
			best_parameters = grid_search.best_estimator_.get_params()
			model = neighbors.KNeighborsClassifier(n_neighbors = DICT_KNN_PARAM['n_neighbors'][n])
			model.fit(x_train, y_train)
					#pdb.set_trace()
			score_test[n] = model.score(x_test, y_test)
			score_train[n] = grid_search.best_score_
			if score_train[n] > best_score:
				best_score = score_train[n]
				best_model = model
				bestn = n
		axs[0].plot(DICT_KNN_PARAM['n_neighbors'],score_train, '.--', linewidth=1)
		axs[1].plot(DICT_KNN_PARAM['n_neighbors'],score_test, '.--', linewidth=1)
		axs[0].plot(DICT_KNN_PARAM['n_neighbors'][bestn],score_train[bestn], 'r*', label = 'best score point', markersize=10)
		axs[0].set_ylabel('training score')
		axs[0].set_xlabel('n_neighbors')
		axs[0].legend(fontsize=6)
		axs[0].grid()
		axs[1].set_ylabel('test score')
		axs[1].set_xlabel('n_neighbors')
		axs[1].legend(fontsize=6)
		axs[1].grid()
		plt.tight_layout()
		plt.savefig('./output/train/knn.png', dpi = 300)
		plt.close('all')
		clf = best_model

	elif 'gbdt' == method:
		clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
	elif 'ann' == method:
		model = MLPClassifier(max_iter=1000, solver='sgd', activation='relu',learning_rate_init=0.1)
		best_score, best_model, bestn, bestm = 0, model, 0, 0
		score_test = np.zeros((len(DICT_ANN_PARAM['layer']), len(DICT_ANN_PARAM['alpha'])))
		score_train = np.zeros((len(DICT_ANN_PARAM['layer']), len(DICT_ANN_PARAM['alpha'])))
		fig, axs = plt.subplots(2,1, figsize = (10, 5))
		for n in range(len(DICT_ANN_PARAM['layer'])):
			for m in range(len(DICT_ANN_PARAM['alpha'])):
				param_grid = {'hidden_layer_sizes': [(100, )* DICT_ANN_PARAM['layer'][n]], 'alpha': [DICT_ANN_PARAM['alpha'][m]]}
				grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=0, cv=5, return_train_score = True)
				grid_search.fit(x_train, y_train)
				best_parameters = grid_search.best_estimator_.get_params()
				model = MLPClassifier(hidden_layer_sizes = (100, )* DICT_ANN_PARAM['layer'][n], alpha = DICT_ANN_PARAM['alpha'][m], max_iter=1000, solver='sgd', activation='relu',learning_rate_init=0.1)
				model.fit(x_train, y_train)
					#pdb.set_trace()
				score_test[n, m] = model.score(x_test, y_test)
				score_train[n, m] = grid_search.best_score_
				if score_train[n, m] > best_score:
					best_score = score_train[n, m]
					best_model = model
					bestn, bestm = n, m
			axs[0].semilogx(DICT_ANN_PARAM['alpha'],score_train[n, :], '.--', label = 'hidden_layer_size = {}x{}'.format(DICT_ANN_PARAM['layer'][n], 100), linewidth=1)
			axs[1].semilogx(DICT_ANN_PARAM['alpha'],score_test[n, :], '.--', label = 'hidden_layer_size = {}x{}'.format(DICT_ANN_PARAM['layer'][n], 100), linewidth=1)
		axs[0].plot(DICT_ANN_PARAM['alpha'][bestm],score_train[bestn, bestm], 'r*', label = 'best score point', markersize=10)
		axs[0].set_ylabel('training score')
		axs[0].set_xlabel('alpha')
		axs[0].legend(fontsize=6)
		axs[0].grid()
		axs[1].set_ylabel('test score')
		axs[1].set_xlabel('alpha')
		axs[1].legend(fontsize=6)
		axs[1].grid()
		plt.tight_layout()
		plt.savefig('./output/train/ann.png', dpi = 300)
		plt.close('all')
		clf = best_model
	else:
		pass

	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	print_output(y_test, y_pred, method, clf.score(x_test, y_test))




def main():
	Y, X, coff = load_data()
	#pdb.set_trace()

	X = preprocessing.scale(X)
	#pdb.set_trace()
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)
	#ml_svr(x_train, x_test, y_train, y_test, coff)
	#ml_adaboost(x_train, x_test, y_train, y_test, coff)
	ml_methods(x_train, x_test, y_train, y_test, coff, 'ann')
	ml_methods(x_train, x_test, y_train, y_test, coff, 'randomforest')
	#ml_methods(x_train, x_test, y_train, y_test, coff, 'adaboost')
	ml_methods(x_train, x_test, y_train, y_test, coff, 'knn')
	ml_methods(x_train, x_test, y_train, y_test, coff, 'svm')
	#ml_methods(x_train, x_test, y_train, y_test, coff, 'gbdt')

if __name__ == '__main__':
	main()