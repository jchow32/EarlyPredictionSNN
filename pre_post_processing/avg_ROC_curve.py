# avg_ROC_curve.py
# plot a ROC curve where the FPR is the x-axis, and the TPR is the y-axis

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
import random


def type_checking(x):
	try:
		return float(x)
	except ValueError:
		return None


def input_file_checking(input_file):
	input_file['FPR'] = input_file['FPR'].apply(type_checking).dropna()
	input_file['TPR'] = input_file['TPR'].apply(type_checking).dropna()
	input_file['FPR_threshold'] = input_file['FPR_threshold'].apply(type_checking).dropna()

	averaged = input_file.groupby(['FPR_threshold']).mean().reset_index().dropna()

	averaged_05 = averaged[(averaged['FPR'] <= 1) & (averaged['FPR'] >= 0.0)]
	averaged_all = averaged[(averaged['FPR'] <= 1) & (averaged['FPR'] >= 0.0)]

	return averaged_05, averaged_all


def rename_to_miss(file_name):
	out_name = file_name.replace("lgd", "miss")
	return out_name


def rename_to_combo(file_name, combo_string):
	if "individual" in file_name:
		out_name = file_name.replace("lgd", combo_string)
	elif "ensemble" not in file_name:
		out_name = file_name.replace("lgd_", "").replace("_fprTPR", "_%s_fprTPR" % combo_string)
	else:
		out_name = file_name.replace("lgd", combo_string)
	return out_name


def select_group(df, labels, model_type):
	selected_group = df[labels]
	selected_group['Model'] = model_type
	selected_group.columns = ['recall', 'precision', 'Model']
	selected_group = selected_group.sort_values(by='recall')
	return selected_group


def get_random(mutation_file):
	mutation_df = pd.read_csv(mutation_file, sep="\t")[['ids', 'PrimaryPhenotype']].drop_duplicates()
	# make some random predictions
	mutation_df['random_prob'] = np.random.uniform(0, 1, mutation_df.shape[0])
	# you should randomly sample a test-set sized portion of sapmles from mutation_df
	mutation_df = mutation_df.sample(frac=0.25)

	c_p, c_r, c_t = precision_recall_curve(mutation_df['PrimaryPhenotype'], mutation_df['random_prob'])
	fpr_tpr = pd.DataFrame({'recall': c_r, 'precision': c_p})
	interested_rates = np.arange(0, 1.01, 0.01)[::-1]
	interested_df = pd.DataFrame(columns=['recall', 'precision'])
	for rate in interested_rates:
		value_01 = fpr_tpr[fpr_tpr['recall'] >= rate].tail(1)  # for PR, that's the largest recall at the rate
		# round FPR down to nearest 0.01, and then groupby the recall (FPR) column
		# for PR, try to find the
		interested_df = interested_df.append(value_01)
	interested_df.index = interested_rates
	interested_df = interested_df.sort_index(ascending=False)
	interested_df['Model'] = "Naive Random"
	auc = np.trapz(interested_df['recall'], x=interested_df['precision'])
	print("Random PR-AUC: %.4f" % auc)

	return interested_df


def group_curve_info(pr):
	pr = pr.groupby('index').mean()

	# you need to convert into long format by concatenating long the recall and precision separately, and adding a label
	# first, select by model
	c_group = select_group(pr, ['c_r', 'c_p'], "SNN")
	r_group = select_group(pr, ['rf_r', 'rf_p'], "Random Forest")
	s_group = select_group(pr, ['s_r', 's_p'], "SVM")
	l_group = select_group(pr, ['l_r', 'l_p'], "Logistic Regression")
	e_group = select_group(pr, ['e_r', 'e_p'], "Ensemble")
	e_ns_group = select_group(pr, ['e_ns_r', 'e_ns_p'], "Ensemble - SNN")

	all_group = pd.concat([c_group, r_group, s_group, l_group, e_group, e_ns_group], axis=0)
	no_ens = pd.concat([c_group, r_group, s_group, l_group], axis=0)

	return all_group, no_ens


def read_pr(pr_info, naive_random, output_string, plot_title):

	if "combo" not in output_string:  # do this for LGD and missense
		mean_pr, df = get_random_pr(naive_random)
	else:  # do this for the max combined
		mean_pr = naive_random
		df = []

	if "roc_roc_concat" not in pr_info:
		print("Naive random avg. for %s: %.4f" % (output_string, mean_pr))

	pr = pd.read_csv(pr_info, sep="\t", header=None)
	# recall then precision in columns
	pr.columns = ['index', 'c_r', 'c_p', 'rf_r', 'rf_p', 's_r', 's_p', 'l_r', 'l_p', 'e_r', 'e_p', 'e_ns_r', 'e_ns_p']

	pr_05 = pr[(pr['index'] <= 1) & (pr['index'] > 0.0)]
	all_group, all_no_ens = group_curve_info(pr)
	all_group_05, all_05_no_ens = group_curve_info(pr_05)

	# all_group.to_csv("%s_testing_file" % pr_info, sep="\t")

	# the only difference between concat of ROC and concat of PR is that ROC is in order FPR, TPR, with FPR on x-axis
	# now you can plot the PR curve using each of the two paired coordinates
	# draw a horizontal line that is the random PR line

	sns.set_style("ticks")
	if "roc_roc_concat" in pr_info:
		plot_roc(all_group, plot_title, "%s_1_ROC.png" % output_string)
		plot_roc(all_no_ens, plot_title, "%s_1_noEnsemble_ROC.png" % output_string)
		plot_roc(all_group_05, plot_title, "%s_05_ROC.png" % output_string)
		plot_roc(all_05_no_ens, plot_title, "%s_05_noEnsemble_ROC.png" % output_string)
	else:
		fig = plt.figure()
		g = sns.lineplot(x='recall', y='precision', data=all_group, hue='Model')
		g.set_xlabel('Recall')
		g.set_ylabel('Precision')
		g.set_title(plot_title)
		g.axhline(mean_pr, ls='--')
		sns.despine()
		plt.savefig('%s_PR.png' % output_string)

		# also plot without ensemble
		fig = plt.figure()
		g = sns.lineplot(x='recall', y='precision', data=all_no_ens, hue='Model')
		g.set_xlabel('Recall')
		g.set_ylabel('Precision')
		g.set_title(plot_title)
		g.axhline(mean_pr, ls='--')
		sns.despine()
		plt.savefig('%s_noEnsemble_PR.png' % output_string)
	return df


def plot_roc(all_group, plot_title, output_string):
	fig = plt.figure()

	# what is the largest TPR (precision) associated with FPR (recall) around 0.06?
	top_6 = all_group[all_group['recall'] < 0.1].sort_values(by='recall', ascending=False)['precision'].iloc[0]

	if "05" in output_string:
		plt.xlim(0, 0.05)
		plt.ylim(0, top_6)  # the y limit depends on the highest y-value associated with the x-value around 0.05

	g = sns.lineplot(x='recall', y='precision', data=all_group, hue='Model')  # this is full range of FPR
	g.set_xlabel('FPR')
	g.set_ylabel('TPR')
	g.set_title(plot_title)
	sns.despine()
	plt.savefig(output_string)


def get_mean_random_pr(df, phenotype_string):
	df = df[[phenotype_string, 'run_id']]
	df = pd.get_dummies(df, columns=[phenotype_string]).groupby('run_id').sum()
	df.columns = ['num_controls', 'num_cases']
	df['num_samples'] = df.loc[:, ['num_controls', 'num_cases']].sum(axis=1)
	df['pr_value'] = df['num_cases'] / df['num_samples']

	mean_pr = df['pr_value'].mean()
	return mean_pr


def get_random_pr(input_file):
	df = pd.read_csv(input_file, sep="\t", header=None)
	df.columns = ['ids', 'probability', 'PrimaryPhenotype', 'run_id']
	out_df = df
	mean_pr = get_mean_random_pr(df, "PrimaryPhenotype")
	return mean_pr, out_df


def load_process_pr(pr_info, out_name, n_random, output_string, miss_string, com_string):
	# read in the precision recall info, for LGD, missense, and combo
	pr_info_miss = rename_to_miss(pr_info)
	pr_info_combo = rename_to_combo(pr_info, out_name)
	n_random_miss = rename_to_miss(n_random)

	lgd_prob = read_pr(pr_info, n_random, output_string, "LGD-specific")
	miss_prob = read_pr(pr_info_miss, n_random_miss, miss_string, "Missense-specific")

	merge_df = pd.merge(lgd_prob, miss_prob, on=['run_id', 'ids'], how='inner')
	combo_mean_random_pr = get_mean_random_pr(merge_df, 'PrimaryPhenotype_x')
	read_pr(pr_info_combo, combo_mean_random_pr, com_string, "Combined")


def main():
	output_string = sys.argv[1]
	pr_info = sys.argv[2]  # looks like *concat  # this is for precision recall
	n_random = sys.argv[3]  # *individualProb file to get the random PR line
	roc_info = sys.argv[4]  # looks like *roc_roc_concat, this is for ROC curve
	out_name = "combo"

	miss_string = rename_to_miss(output_string)
	com_string = output_string.replace("lgd_", "%s_" % out_name)

	# Plotting precision curve from pr_info concat file:
	print("Plotting PR curves")
	load_process_pr(pr_info, out_name, n_random, output_string, miss_string, com_string)

	# Plotting the ROC curve in a similar way to the precision curve:
	print("Plotting ROC curves")
	load_process_pr(roc_info, out_name, n_random, output_string, miss_string, com_string)


if __name__ == "__main__":
	main()
