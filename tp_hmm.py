from __future__ import print_function

from hmmlearn import hmm
import os
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DATA_DIR = './trace/'
DST_DIR = './hmm_sample/'
MIX_DST_DIR = './mix_sample/'
NEW_MIX_DST_DIR = './new_mix_sample/'
NEW_MIX_DST_DIR_FORMAT = './new_mix_format_sample/'

DATA_LEN = 600	# half second
data_file_dir = os.listdir(DATA_DIR)
data_list = []
model_list = []
UPPER_BOUND = 900.0
LOWER_BOUND = 0.0
TRACE_NUM = 3
SAVE = 0
SHOW_FIG = 0

SAVE_MIX = 1
SHOW_MIX = 0
TRANSIT_1 = [0.70, 0.15, 0.15]
TRANSIT_2 = [0.70, 0.15, 0.15]
TRANSIT_5 = [0.70, 0.15, 0.15]


NEW_SAVE_MIX = 1
NEW_SHOW_MIX = 0
NEW_TRANSIT_1 = [0.90, 0.05, 0.05]
NEW_TRANSIT_2 = [0.05, 0.90, 0.05]
NEW_TRANSIT_5 = [0.05, 0.05, 0.90]


DURATION_LOW = 20
DURATION_HIGH = 100

NEW_DURATION_LOW = 5
NEW_DURATION_HIGH = 10

def hmm_modeling(data, n_com, lengths = []):
	model = hmm.GaussianHMM(n_components=n_com, covariance_type="full", n_iter=3000)
	# model.transmat_= np.array([[0.7, 0.1, 0.1, 0.1],
	# 							[0.6, 0.2, 0.1, 0.1],
	# 							[0.6, 0.1, 0.2, 0.1],
	# 							[0.6, 0.1, 0.1, 0.2]])
	if lengths == []:
		model.fit(data)
	else:
		model.fit(data, lengths)
	# hidden_states = model.predict(new_data)

	prob_sum = 0.0
	for i in range(len(model.startprob_)-1):
		model.startprob_[i] = 1.0/n_com
		prob_sum += 1.0/n_com
	model.startprob_[-1] = 1 - prob_sum
	# print("Transition matrix")
	# print(model.transmat_)
	# print(model.startprob_)
	# print()


	# print("Means and vars of each hidden state")
	for i in range(model.n_components):
		# print("{0}th hidden state".format(i))
		# print("mean = ", model.means_[i])
		# print("var = ", np.diag(model.covars_[i]))
		pass
	return model


def hmm_sampling(model_pair, n_traces = 1, trace_length = 600):
	model_name = model_pair[0]
	model = model_pair[1]
	sampling = [model_name]
	for i in range(n_traces):
		X,Y = model.sample(trace_length)
		X = [min(max(LOWER_BOUND, x), UPPER_BOUND) for x in X]
		sampling.append([X, 'MIX_' + str(i)])
	return sampling

def plot_figure(data, index, trace_length = 600):
	p = plt.figure(index, figsize=(20,5))
	plt.plot(range(trace_length), data, color='cornflowerblue', label='BW Trace ' + str(index), linewidth=2.5)
	plt.legend(loc='upper right',fontsize=20)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Mbps', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	# plt.axis([0, trace_length, 0, 1200])
	plt.xticks(np.arange(0, trace_length+1, 50))
	plt.yticks(np.arange(200, 100*(int(max(data)/100)+4) + 1, 200))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085,right=0.97)	
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	return p

def save_data(model_list):
	figures = []
	for i in range(len(model_list)):
		samples = hmm_sampling(model_list[i], TRACE_NUM)
		model_name = samples[0]
		for j in range(1, len(samples)):
			np.savetxt(DST_DIR + model_name + '_' + str(samples[j][1]) + '.txt', samples[j][0], fmt='%1.2f')
			if SHOW_FIG:
				figures.append(plot_figure(samples[j][0], i*TRACE_NUM + j))
# overall_list = hmm_sampling(model_list[-1], 10)
	if SHOW_FIG:
		for p in figures:
			p.show()
		raw_input()

def new_generate_1_2_5(model_list, n_traces = 1):
	new_model_list = []
	start_model_idx = 0
	samples = []
	transmatrix = []
	model_record = []
	for model_pair in model_list:
		if model_pair[0] in ['HMM_1', 'HMM_2', 'HMM_5']:
			new_model_list.append(model_pair)

	transmatrix.append(NEW_TRANSIT_1)
	transmatrix.append(NEW_TRANSIT_2)
	transmatrix.append(NEW_TRANSIT_5)

	for i in range(n_traces):
		temp_model_idx = start_model_idx
		temp_model = new_model_list[start_model_idx][1]
		temp_sample = []

		while len(temp_sample) < DATA_LEN:
			sample_len = np.random.randint(NEW_DURATION_LOW, NEW_DURATION_HIGH)
			if len(temp_sample) + sample_len > DATA_LEN:
				sample_len = DATA_LEN - len(temp_sample)
			X,Y = temp_model.sample(sample_len)
			X_new = [float(min(max(LOWER_BOUND, x), UPPER_BOUND)) for x in X]
			temp_sample = temp_sample + X_new
			model_record.append(temp_model_idx)
			temp_model_idx = transit(transmatrix, temp_model_idx)
			temp_model = new_model_list[temp_model_idx][1]
		samples.append([temp_sample, i, model_record])

	if NEW_SAVE_MIX:
		for i in range(len(samples)):
			curr_time = 0.0
			model_name = samples[i][1]
			record = samples[i][2]
			# print(record)
			save_sample = samples[i][0][:]

			for j in range(len(save_sample)):
				temp_tp = save_sample[j]
				save_sample[j] = [curr_time, temp_tp]
				curr_time += 0.5
			np.savetxt(NEW_MIX_DST_DIR + 'NEW_MIX_'+ str(model_name) + '.txt', samples[i][0], fmt='%1.2f')
			np.savetxt(NEW_MIX_DST_DIR_FORMAT + 'NEW_MIX_FORMAT_'+ str(model_name) + '.txt', save_sample, fmt='%1.2f')

			if SHOW_MIX == 1:
				figures.append(plot_figure(samples[i][0], i))

		if NEW_SHOW_MIX:
			for p in figures:
				p.show()
			raw_input()
	return

def generate_1_2_5(model_list, n_traces = 1):
	new_model_list = []
	transmatrix = []
	start_model_idx = 0
	figures = []
	samples = []

	for model_pair in model_list:
		if model_pair[0] in ['HMM_1', 'HMM_2', 'HMM_5']:
			new_model_list.append(model_pair)

	assert len(new_model_list) == 3
	for new_model in new_model_list:
		if new_model[0].split('_')[1] == 1:
			start_model_idx = len(transmatrix)	
			transmatrix.append(TRANSIT_1)
		elif new_model[0].split('_')[1] == 2:
			transmatrix.append(TRANSIT_2)
		else:
			transmatrix.append(TRANSIT_5)

	for i in range(n_traces):
		temp_time = 0
		total_time = 0
		temp_model_idx = start_model_idx
		temp_model = new_model_list[start_model_idx][1]
		temp_sample = []
		model_record = []

		while total_time < DATA_LEN:
			temp_time = np.random.randint(DURATION_LOW, DURATION_HIGH)
			if temp_time + total_time > DATA_LEN:
				temp_time = DATA_LEN - total_time
			total_time += temp_time
			model_record.append([temp_model_idx, temp_time])
			X,Y = temp_model.sample(temp_time)
			X = [float(min(max(LOWER_BOUND, x), UPPER_BOUND)) for x in X]
			temp_sample = temp_sample + X
			temp_model_idx = transit(transmatrix, temp_model_idx)
			temp_model = new_model_list[temp_model_idx][1]
		samples.append([temp_sample, i, model_record])

	if SAVE_MIX:
		for i in range(len(samples)):
			model_name = samples[i][1]
			record = samples[i][2]
			np.savetxt(MIX_DST_DIR + 'MIX_'+ str(model_name) + '.txt', samples[i][0], fmt='%1.2f')
			if SHOW_MIX == 1:
				figures.append(plot_figure(samples[i][0], i))

		if SHOW_MIX:
			for p in figures:
				p.show()
			raw_input()
	return

def transit(transmatrix, current_idx):
	assert current_idx < len(transmatrix)
	prob = transmatrix[current_idx]
	prob_cumsum = np.cumsum(prob)
	random_num = np.random.uniform()
	# print(prob_cumsum, random_num)
	ret_idx = (prob_cumsum > random_num).argmax()
	# print(ret_idx)
	return ret_idx

def main():
	all_sample = []
	for data_file in data_file_dir:
		data_path = DATA_DIR + data_file
		data = []
		with open(data_path, 'rb') as f:
			for line in f:
				parse = [float(line.rstrip('\n'))]
				data.append(parse)
		new_data = data[:DATA_LEN]
		data_list.append(new_data)
		# print(data_path)
		data_std = np.std(new_data)
		data_mean = np.mean(new_data)
		re_std = data_std/data_mean
		# print(re_std)

		if re_std < 0.1:
			n_com = 1
		elif re_std < 0.4:
			n_com = 4
		else:
			n_com = 5

		model = hmm_modeling(new_data, n_com)
		file_index = data_file.split('.')[0].split('_')[1]
		model_list.append(['HMM_' + file_index, model])

	# # For overall data
	# overall_data = np.concatenate(data_list)
	# lengths = [DATA_LEN] * len(data_list)

	# # For independent
	# overall_model = hmm_modeling(overall_data, 3, lengths)
	# # For overall concatenate
	# model_list.append(['overall', overall_model])

	# Higher level transit_mat, for certain period and then transit
	# generate_1_2_5(model_list, n_traces = 5)

	# Higher level transit_mat, transit at each point
	new_generate_1_2_5(model_list, n_traces = 40)



	if SAVE:
		save_data(model_list)


if __name__ == '__main__':
	main()