import pandas as pd
import pdb
import datetime;
import numpy as np
import pydotplus
import pickle
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn import preprocessing, neighbors, tree
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from ml_method import ml_methods
PLAYS = [1, 10, 100, 100000]
CATAGORIES = ['NewsClip', 'Animation', 'Lecture']
ENCODERS = ['libx264', 'hevc', 'vp8', 'vp9']
def read_power_data(power_data_file):
	df = pd.read_csv(power_data_file)
	#df['last_changed'] = pd.to_datetime(df['last_changed'], errors='coerce', utc=None).tz_localize(None)
	df['last_changed'] = df.apply(lambda x:  pd.to_datetime(x.last_changed).tz_localize(None) + datetime.timedelta(seconds = 3600), axis=1)
	return df

def str2float(strin):
	if 'k' or 'K' in strin:
		#print (strin)
		ret = float(strin.strip('k').strip('K'))*1024
	else:
		ret = float(strin)
	return ret
	#if 'm' or 'M' in strin:
		#ret = float(strin.strip('m').strip('M')*1024*1024)	

def read_video_data(video_data_file, metric_data_file, df_pdata):
	data0 = pd.read_csv('out_0.csv')
	data1 = pd.read_csv('out.csv')
	data0 = pd.concat([data0, data1], axis = 0, ignore_index = True)
	data_metric = pd.read_csv(metric_data_file)
	data0['enc_t0'] = pd.to_datetime(data0['enc_t0'], errors='coerce')
	data0['enc_t1'] = pd.to_datetime(data0['enc_t1'], errors='coerce')
	data0['play_t0'] = pd.to_datetime(data0['play_t0'], errors='coerce')
	data0['play_t1'] = pd.to_datetime(data0['play_t1'], errors='coerce')
	data0['duration_enc(s)'] = (data0['enc_t1'] - data0['enc_t0']).dt.total_seconds()
	data0['duration_play(s)'] = (data0['play_t1'] - data0['play_t0']).dt.total_seconds()
	data0['power_enc(W)'], data0['power_play(W)'], data0['max_power_enc(W)'], data0['max_power_play(W)'], data0['resolutionx'], data0['resolutiony'] = 0.0,0.0,0.0,0.0,0.0,0.0
	data0['spatial_complexity'], data0['temporal_complexity'], data0['color_complexity'], data0['chunk_complexity_variation'] = 0.0, 0.0, 0.0, 0.0
	#pdb.set_trace()
	BASIC_POWER = 81.306

	for index, row in data0.iterrows():
		#try:
		if(1):

			enc_t0, enc_t1, play_t0, play_t1 = row['enc_t0'], row['enc_t1'], row['play_t0'], row['play_t1']
			time_delta = datetime.timedelta(seconds = 4)
			#pdb.set_trace()
			#if power_slice_enc.empty:
			power_slice_enc = df_pdata[(df_pdata['last_changed'] > enc_t0 - time_delta) & (df_pdata['last_changed'] < enc_t1 + time_delta)]
			power_slice_enc = power_slice_enc[power_slice_enc['power_consumption'] > 90]
			if power_slice_enc.empty:
				power_slice_enc = df_pdata[(df_pdata['last_changed'] > enc_t0 - time_delta) & (df_pdata['last_changed'] < enc_t1 + time_delta)]
				#print(power_slice_enc)
				power_slice_enc = power_slice_enc.sort_values(by='power_consumption', ascending=False).head(1)


			power_slice_play = df_pdata[(df_pdata['last_changed'] > play_t0) & (df_pdata['last_changed'] < play_t1)]
			#pdb.set_trace()
			#if power_slice_play.empty:
				#power_slice_play = df_pdata[(df_pdata['last_changed'] > play_t0) & (df_pdata['last_changed'] < play_t1)]
			power_enc = power_slice_enc.mean()['power_consumption'] - BASIC_POWER
			power_play = power_slice_play.mean()['power_consumption'] - BASIC_POWER
			data0['power_enc(W)'][index] = power_enc
			data0['power_play(W)'][index] = power_play
			power_enc = power_slice_enc.max()['power_consumption'] - BASIC_POWER
			power_play = power_slice_play.max()['power_consumption'] - BASIC_POWER
			data0['max_power_enc(W)'][index] = power_enc
			data0['max_power_play(W)'][index] = power_play		
			data0['tbn'][index] = str2float(data0['tbn'][index])
			data0['tbc'][index] = str2float(data0['tbc'][index])
			data0['size0'][index] = str2float(data0['size0'][index])
			data0['size_enc'][index] = str2float(data0['size_enc'][index])
			resolution = data0['resolution'][index]
			h,w = resolution.split('x')
			data0['resolutionx'][index] = int(h)
			data0['resolutiony'][index] = int(w)
			data0['resolution'][index] = data0['path'][index].split('/')[-2]

			videoname = row['file name'].strip('.mkv')
			metric = data_metric[data_metric['FILENAME'] == videoname]
			if  metric.empty:
				data0['size_enc'][index] = 0
			else:
				data0['spatial_complexity'][index] = metric['SPATIAL_COMPLEXITY']
				data0['temporal_complexity'][index] = metric['TEMPORAL_COMPLEXITY']
				data0['color_complexity'][index] = metric['COLOR_COMPLEXITY']
				data0['chunk_complexity_variation'][index] = metric['CHUNK_COMPLEXITY_VARIATION']

			#pdb.set_trace()
		#except:
			#print('Bade case')
			#print(data0.loc[index].to_frame().T)
			#data0['size_enc'][index] = 0

	data0 = data0[data0['size_enc'] != 0]
	data0['Energy_enc(J)']   = data0['power_enc(W)']*data0['duration_enc(s)']
	data0['Energy_play(J)']  = data0['power_play(W)']*20 #for one play #data0['duration_play(s)']/4

	data0['Energy_trans(J)'] = 0.05*data0['size_enc']/1024/1024/1204*3600
	data0['Energy(J)'] = data0['Energy_enc(J)'] + data0['Energy_play(J)'] + data0['Energy_trans(J)']
	data0['cmprs_ratio'] = data0['size_enc']/data0['size0']
	ret = data0.copy()
	for play_num in PLAYS[1:]:
		datax = data0.copy()
		datax['plays'] = play_num;
		datax['Energy_play(J)']  = datax['Energy_play(J)']*play_num
		datax['Energy_trans(J)'] = datax['Energy_trans(J)']*play_num
		datax['Energy(J)'] = datax['Energy_enc(J)'] + datax['Energy_play(J)'] + datax['Energy_trans(J)']
		ret = pd.concat([ret, datax], axis = 0, ignore_index = True)
	#encoder_1hot = pd.get_dummies(data0['encoder'])
	#data0 = pd.concat([data0,pd.get_dummies(data0['encoder']) ], axis=1)

	return ret
def plot_power_time(df_vdata, df_pdata):
	#pdb.set_trace()
	time_delta = datetime.timedelta(seconds = 10)
	t0, t1 = df_vdata['enc_t0'][0] - time_delta, df_vdata['play_t1'][3]
	df_pdata[(df_pdata['last_changed'] > t0) & (df_pdata['last_changed'] < t1)].plot(x = 'last_changed', y = 'power_consumption')
	plt.ylabel('power(W)')
	plt.xlabel('time(hh mm:ss)')
	plt.grid(axis = 'both')
	plt.savefig('./output/busy_state.png', dpi = 300)
	plt.close('all')

	df_pdata.loc[500:1000].plot(x = 'last_changed', y = 'power_consumption')
	plt.ylabel('power(W)')
	plt.xlabel('time(hh mm:ss)')
	plt.ylim((75, 230))
	plt.grid(axis = 'both')
	plt.savefig('./output/idle_state.png', dpi = 300)
	plt.close('all')

def plot_power_stat(df_vdata):
	data0 = df_vdata[df_vdata['plays'] == 1]
	
	for encoder in ENCODERS:
		fig, axs = plt.subplots(1, 2, figsize = (20, 10))
		data1 = data0[data0['encoder'] == encoder]
		x0 = np.array(data1['max_power_enc(W)'])
		axs[0].hist(x0, bins = 40)
		axs[0].set_ylabel('frequency', fontsize = 15)
		axs[0].set_title('maximum power distribution of encoding', fontsize = 15)
		axs[0].set_xlabel('encoding power (W)', fontsize = 15)

		x1 = np.array(data1['max_power_play(W)'])
		axs[1].hist(x1, bins = 40)
		axs[1].set_ylabel('frequency', fontsize = 15)
		axs[1].set_title('maximum power distribution of decoding & play', fontsize = 15)
		axs[1].set_xlabel('decoding & play power (W)', fontsize = 15)
		plt.tight_layout()
		plt.savefig('./output/maximum_power_distribution_encoder_{}.png'.format(encoder), dpi = 300)
		plt.close('all')
		print('encoding: encoder = {}, mean = {}, std = {}, maximum = {}'.format(encoder, np.mean(x0), np.std(x0), np.max(x0)))
		print('decoding: encoder = {}, mean = {}, std = {}, maximum = {}'.format(encoder, np.mean(x1), np.std(x1), np.max(x1)))

	data0 = df_vdata[(df_vdata['plays'] == 1) & (df_vdata['resolution'] == '1080P')]
	for encoder in ENCODERS:
		fig, axs = plt.subplots(1, 3, figsize = (20, 10))
		data1 = data0[data0['encoder'] == encoder]
		x0 = np.array(data1['Energy_enc(J)'])
		axs[0].hist(x0, bins = 40)
		axs[0].set_ylabel('frequency', fontsize = 15)
		axs[0].set_title('energy consumption distribution of encoding', fontsize = 15)
		axs[0].set_xlabel('encoding Energy consumption(J)', fontsize = 15)

		x1 = np.array(data1['Energy_trans(J)'])
		axs[1].hist(x1, bins = 40)
		axs[1].set_ylabel('frequency', fontsize = 15)
		axs[1].set_title('energy consumption distribution of transmission', fontsize = 15)
		axs[1].set_xlabel('transmission energy consumption(J)', fontsize = 15)

		x2 = np.array(data1['Energy_play(J)'])
		axs[2].hist(x2, bins = 40)
		axs[2].set_ylabel('frequency', fontsize = 15)
		axs[2].set_title('energy consumption distribution of decoding & play', fontsize = 15)
		axs[2].set_xlabel('decoding & play energy consumption(J)', fontsize = 15)


		plt.tight_layout()
		plt.savefig('./output/mean_power_distribution_encoder_{}.png'.format(encoder), dpi = 300)
		plt.close('all')
		print('================================')
		print('encoding: encoder = {}, mean = {}, std = {}, maximum = {}'.format(encoder, np.mean(x0), np.std(x0), np.max(x0)))
		print('trans:    encoder = {}, mean = {}, std = {}, maximum = {}'.format(encoder, np.mean(x1), np.std(x1), np.max(x1)))
		print('decoding: encoder = {}, mean = {}, std = {}, maximum = {}'.format(encoder, np.mean(x2), np.std(x2), np.max(x2)))

	#pdb.set_trace()



def bar_plot(data1, output_name):
		engy_enc = np.array(data1['Energy_enc(J)'])
		engy_trans = np.array(data1['Energy_trans(J)'])
		engy_play = np.array(data1['Energy_play(J)'])
		xticks = data1['encoder'].tolist()
		x = [ i for i in range(len(xticks))]
		fig, ax = plt.subplots(1,1, figsize = (20, 10))
		plt.xticks(x, xticks, rotation=45)
		plt.tick_params(axis = 'x', labelsize=20)
		plt.tick_params(axis = 'y', labelsize=20)
		ax.bar(x, engy_enc, label = 'Energy of encoding (J)')
		ax.bar(x, engy_trans, bottom = engy_enc, label = 'Energy of transmission (J)')
		ax.bar(x, engy_play,  bottom = engy_enc + engy_trans, label = 'Energy of decoding & play (J)')
		
		plt.grid(axis = 'both')
		plt.legend(fontsize=20)
		plt.xlabel('Encoding format', fontsize = 20)
		plt.ylabel('Average energy consumption (J)', fontsize = 20)
		plt.tight_layout()
		plt.savefig('./output/{}'.format(output_name), dpi = 300)
		plt.close('all')

def point_plot(df_vdata, x = 'size0', y = 'Energy(J)', ylabel = 'Average energy consumption (J)'):
	data03 = df_vdata.groupby(['plays', x, 'encoder']).mean().reset_index()
	for play in set(data03['plays'].tolist()):
		fig, ax = plt.subplots(1,1, figsize = (20, 10))
		for encoder in ENCODERS:
			temp = data03[(data03['encoder'] == encoder) & (data03['plays'] == play)]
			xx = temp[x]/1024/1024
			yy = temp[y]
			ax.loglog(xx, yy, '.' ,label = encoder)
		plt.tick_params(axis = 'x', labelsize=20)
		plt.tick_params(axis = 'y', labelsize=20)
		plt.grid(axis = 'both')
		plt.legend(fontsize=20)
		plt.xlabel('File size(MB)', fontsize = 20)
		plt.ylabel(ylabel, fontsize = 20)
		plt.tight_layout()
		plt.savefig('./output/{}_vs_{}_play{}_for_different_encoder.png'.format(y, x, play), dpi = 300)
		plt.close('all')
def plot_energy_consumption(df_vdata):
	#df_vdata['size0'] = np.log10(np.array(df_vdata['size0'])).astype('int64')
	data0 = df_vdata.groupby(['plays', 'encoder']).mean().reset_index()
	data02 = df_vdata[df_vdata['resolution'] == '1080P'].groupby(['plays', 'video type', 'encoder']).mean().reset_index()
	data03 = df_vdata.groupby(['plays', 'resolution', 'encoder']).mean().reset_index()

	
	for play in set(data0['plays'].tolist()):
		data1 = data0[(data0['plays'] == play)]
		bar_plot(data1, 'energy_vs_encoder_play{}.png'.format(play))
		for cat in set(data02['video type'].tolist()):
			data2 = data02[(data02['video type'] == cat) & (data02['plays'] == play)]
			bar_plot(data2, 'energy_vs_encoder_play{}_category_{}.png'.format(play, cat))
		
		for res in set(data03['resolution'].tolist()):
			data3 = data03[(data03['resolution'] == res) & (data03['plays'] == play)]
			bar_plot(data3, 'energy_vs_encoder_play{}_resolution_{}.png'.format(play, res))
		
	point_plot(df_vdata)
	point_plot(df_vdata, y = 'Energy_enc(J)')
	point_plot(df_vdata, y = 'Energy_trans(J)')
	point_plot(df_vdata, y = 'Energy_play(J)')


	point_plot(df_vdata, y = 'cmprs_ratio', ylabel = 'compression ratio')
	point_plot(df_vdata, y = 'duration_enc(s)', ylabel = 'average elapsed time (s)')



def explore_video_data(df_vdata):
	plot_energy_consumption(df_vdata)

def train_test_model_power_regress(df_vdata):
	data = df_vdata.copy()
	encoder_1hot = pd.get_dummies(data0['encoder', 'video type'])
	data = pd.concat([data0,  encoder_1hot], axis=1)
	feat_name = ['bitrate', 'resolutionx', 'resolutiony', 'fps', 'tbn', 'tbc', 'cmprs_ratio', 'duration_enc(s)', 'duration_play(s)', 'plays'] + encoder_1hot.columns.tolist()
	label_name = ['Energy(J)']
	X, Y = data[feat_name], data[label_name]
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)
	min_max_scaler = preprocessing.MinMaxScaler()
	x_train = min_max_scaler.fit_transform(x_train)
	x_test  = min_max_scaler.transform(x_test)
	y_train = np.array(y_train)[:, 0]
	y_test = np.array(y_test)[:, 0]

	# svr
	svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
	svr.fit(x_train, y_train)
	y_pred = svr.predict(x_test)
	score = svr.score(x_test, y_test)

	# random forest
	rf = GridSearchCV(RandomForestRegressor(n_estimators=10, random_state=0), cv=5,param_grid={"n_estimators": [1, 10, 100, 1000, 10000]})
	rf.fit(x_train, y_train)
	score1 = rf.score(x_test, y_test)

	svm = GridSearchCV(SVC(), cv = 5, param_grid = {'C':[0.1, 1,10],'gamma':[10, 1, 0.001], 'kernel':['rbf']}, verbose=1, n_jobs = 10);
	svm.fit(x_train, y_train)
	score_svm = svm.score(x_test, y_test)
	predict = svm.predict(x_test)
	print('svm score : ', score_svm)
	print('svm params :', svm.best_params_)
	print('svm report: ', classification_report(y_test,predict))

	rf = GridSearchCV(RandomForestClassifier(n_estimators=10, random_state=0), cv=5,param_grid={"n_estimators": [1, 10, 100, 1000, 5000]}, verbose=1, n_jobs = 10)
	rf.fit(x_train, y_train)
	predict = rf.predict(x_test)
	score_rf = rf.score(x_test, y_test)
	print('rf score : ', score_rf)
	print('rf params :', rf.best_params_)
	print('rf report: ', classification_report(y_test,predict))

	pdb.set_trace()
	return rf, min_max_scaler

def train_test_model_codec_class(df_vdata):

	data = df_vdata.copy()
	data = df_vdata.sort_values(by = 'Energy(J)').groupby(['path', 'file name', 'plays']).head(1)
	encoder_1hot = pd.get_dummies(data['video type'])
	data = pd.concat([data,  encoder_1hot], axis=1)
	feat_name = ['plays','size0', 'resolutionx', 'resolutiony', 'fps', 'spatial_complexity', 'temporal_complexity', 'color_complexity','chunk_complexity_variation'] + encoder_1hot.columns.tolist()
	label_name = ['encoder']
	X, Y = np.array(data[feat_name].astype(float)), np.array(data[label_name])
	X[:, 0:2] = np.log10(X[:, 0:2])
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)
	for label in ENCODERS:
		print('label: {}, train num = {}'.format(label, np.sum(y_train == label)))
		print('label: {}, test num = {}'.format(label, np.sum(y_test == label)))

	min_max_scaler = preprocessing.MinMaxScaler()
	x_train = min_max_scaler.fit_transform(x_train)
	x_test  = min_max_scaler.transform(x_test)
	y_train = np.array(y_train)[:, 0]
	y_test = np.array(y_test)[:, 0]

	coff = None
	#ml_methods(x_train, x_test, y_train, y_test, coff, 'ann')
	ml_methods(x_train, x_test, y_train, y_test, coff, 'randomforest')
	ml_methods(x_train, x_test, y_train, y_test, coff, 'knn')
	ml_methods(x_train, x_test, y_train, y_test, coff, 'svm')


	'''randome forest is reported as the best perf model'''

	model = RandomForestClassifier(n_estimators = 200, max_depth = 150)
	model.fit(x_train, y_train)

	importances = model.feature_importances_
	feat_name[0] = 'requests'
	feat_name[1] = 'video size' 
	feat_name[2] = 'width'
	feat_name[3] = 'height' 
	feat_name[8] = 'chunk_variation' 
	x_columns = feat_name
	x_columns_indices = []
	indices = np.argsort(importances)[::-1]
	for f in range(x_train.shape[1]):
		x_columns_indices.append(feat_name[indices[f]])
	x = [i for i in range(len(feat_name))]
	fig, ax = plt.subplots(1,1, figsize = (10, 5))
	ax.bar(x, importances[indices], color='orange', align='center')
	plt.xticks(x, x_columns_indices, rotation=90, fontsize=10)
	plt.tight_layout()
	plt.ylabel('importance level')
	plt.xlabel('feature name')
	plt.savefig('feature_importance.png', dpi = 500)
	plt.close('all')

	'''
	for index, model in enumerate(model.estimators_):
		filename = 'rf_' + str(index) + '.pdf'
		dot_data = tree.export_graphviz(model, out_file=None, feature_names=feat_name,class_names=ENCODERS,filled=True, rounded=True,special_characters=True)
		graph = pydotplus.graph_from_dot_data(dot_data)
		graph.write_pdf(filename)
	'''

	return model, min_max_scaler, feat_name


def inference(feature, model, scalar):
	x_inf = scalar.transform(feature)
	y_predict = model.predict(x_inf)
	print("The best encoding format is: {}".format(y_predict))

def main():
	
	power_data_file='power.csv'
	video_data_file='out.csv'
	metric_data_file = 'original_videos_metrics1.csv'
	df_pdata = read_power_data(power_data_file)
	df_vdata = read_video_data(video_data_file, metric_data_file, df_pdata)
	plot_power_time(df_vdata, df_pdata)
	df_vdata.to_csv('df_vdata_tmp.csv')
	
	df_vdata = pd.read_csv('df_vdata_tmp.csv')
	plot_power_stat(df_vdata)
	explore_video_data(df_vdata)
	model, scalar, feat_name = train_test_model_codec_class(df_vdata)
	#pickle.dump([model, scalar, feat_name], open('model.sav', 'wb'))

if __name__ == '__main__':
	main()