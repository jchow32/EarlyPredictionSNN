# test_cases.py
# from the predicted probabilities of artificial test cases which have a single LGD mutation
# 1) find the correlation between predicted probability and pLI for that gene
# 2) determine enrichment of de novo LGD and missense in cases relative to controls

import sys
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
import seaborn as sns
import numpy as np


def type_checking(x):
	try:
		return float(x)
	except ValueError:
		return None


def process_prob_file(probability_file, id_file, input_name, sfari_scores):
	probability_file = pd.read_csv(probability_file, header=None, error_bad_lines=False, sep="\t", names=[
		'ids', 'probability', 'phenotype', 'identifier']).dropna().drop(['identifier'], axis=1)
	probability_file['probability'] = probability_file['probability'].apply(type_checking).dropna()
	probability_file = probability_file[['ids', 'probability']]

	# get the mean probability per id
	probability_file = probability_file.groupby(['ids']).mean()
	id_file = pd.read_csv(id_file, sep="\t")
	probability_file = pd.merge(id_file, probability_file, on='ids', how='inner')

	out_file = pd.merge(probability_file, sfari_scores, on='gene', how='left')
	out_file['probability'] = out_file['probability'].round(5)
	out_file.sort_values(by='probability', ascending=False).to_csv("%s_sorted" % input_name, sep="\t", index=False)
	return probability_file


def calculate_corr(df, field):
	pearson = pearsonr(df[field], df['probability'])
	spearman = spearmanr(df[field], df['probability'], nan_policy='omit', axis=None)

	print(pearson)
	print(spearman)

	round_pearson = round(pearson[0], 4)
	round_spearman = round(spearman[0], 4)
	round_pear_p = '{:0.3e}'.format(round(pearson[1]))
	round_spear_p = '{:0.3e}'.format(round(spearman[1]))
	return round_pearson, round_spearman, round_pear_p, round_spear_p


def select_rename(df, mutation_type, total_case, total_control):
	case_df = df[(df['PrimaryPhenotype'] == 1) & (df['LOF'] == mutation_type)][['gene', 'count']]
	case_df.columns = ['gene', 'case_count']
	case_df['case_count'] = case_df['case_count'] / total_case

	control_df = df[(df['PrimaryPhenotype'] == 0) & (df['LOF'] == mutation_type)][['gene', 'count']]
	control_df.columns = ['gene', 'control_count']
	control_df['control_count'] = control_df['control_count'] / total_control

	merge_df = pd.merge(case_df, control_df, on='gene', how='outer')
	merge_df = merge_df.fillna(0)
	diff_string = "%s_norm_diff" % mutation_type
	merge_df[diff_string] = merge_df['case_count'] - merge_df['control_count']

	return merge_df[['gene', diff_string]]


def create_plot(complete_df, joined, input_file_name):
	bins = np.arange(0, 1.05, 0.05).tolist()
	labels = [
		'0-0.05', '0.05-0.1', '0.1-0.15', '0.15-0.20', '0.20-0.25', '0.25-0.30', '0.30-0.35', '0.35-0.40',
		'0.40-0.45', '0.45-0.50', '0.50-0.55', '0.55-0.60', '0.60-0.65', '0.65-0.70', '0.70-0.75', '0.75-0.80',
		'0.80-0.85', '0.85-0.90', '0.90-0.95', '0.95-1.0']
	complete_df['binned'] = pd.cut(complete_df['probability'], bins, labels=labels)

	plt.figure()
	sns.set_style("ticks")
	grey_colors = ['lightgrey' for x in labels]

	fig, ax = plt.subplots(2, 2)
	ax = ax.flatten()
	g1 = sns.barplot(data=complete_df, x="binned", y="LGD_norm_diff", alpha=0.5, ax=ax[0], palette=grey_colors)
	g1.set(xlabel='Probability', ylabel='Difference in LGD enrichment (normalized)')  # this is case - control: norm LGD
	for item in g1.get_xticklabels():
		item.set_rotation(90)

	g2 = sns.barplot(data=complete_df, x="binned", y="missense_norm_diff", alpha=0.5, ax=ax[1], palette=grey_colors)
	g2.set(xlabel='Probability', ylabel='Difference in missense enrichment (normalized)')
	for item in g2.get_xticklabels():
		item.set_rotation(90)

	print("Getting Spearman and Pearson correlations")
	print("Correlation with pLI")
	pli_pear, pli_spear, pli_pear_p, pli_spear_p = calculate_corr(joined.dropna(axis=0), 'pLI')
	print("Correlation with LOEUF")
	l_pear, l_spear, l_pear_p, l_spear_p = calculate_corr(joined.dropna(axis=0), 'oe_lof_upper')

	p3 = sns.regplot(
		x="probability", y="pLI", data=joined, ax=ax[2], color='b', ci=None, fit_reg=False, x_ci=None,
		scatter_kws={'alpha': 0.3})
	p3.text(
		0, joined['pLI'].max() + 0.05, "Pearson: %s, Spearman: %s" % (pli_pear, pli_spear), color='black')
	ax[2].set(xlabel='Probability')

	p3 = sns.regplot(
		x="probability", y="oe_lof_upper", data=joined, ax=ax[3], color='b', ci=None, fit_reg=False, x_ci=None,
		scatter_kws={'alpha': 0.3})
	p3.text(
		0, joined['oe_lof_upper'].max() + 0.05, "Pearson: %s, Spearman: %s" % (l_pear, l_spear), color='black')
	ax[3].set(xlabel='Probability', ylabel='LOEUF')

	sns.despine()
	plt.savefig("%s_enrich_bar.png" % input_file_name)
	plt.savefig("%s_enrich_bar.pdf" % input_file_name)


def full_processing(input_file, pli_file, sfari_scores, mutation_df, file_name):
	# join together the pLI and the SFARI scores with the average probability
	joined = pd.merge(input_file, pli_file, on='gene', how='left')
	joined = pd.merge(joined, sfari_scores, on='gene', how='left')
	joined['score'] = joined['score'].fillna("None")

	# here you can also join the number of de novo LGD mutations per gene from denovo-db
	total_control = 1911 + 250 + 84
	total_case = 2508 + 1445 + 1625 + 10 + 30 + 51 + 4293

	count_mutation = (mutation_df[['gene', 'LOF', 'PrimaryPhenotype']].groupby(
		by=['gene', 'PrimaryPhenotype', 'LOF']).size().reset_index(name='count'))
	# now get the case and control separately, and then horizontally (wide) concatenate them
	miss_df = select_rename(count_mutation, 'missense', total_case, total_control)
	lgd_df = select_rename(count_mutation, 'LGD', total_case, total_control)
	both_df = pd.merge(lgd_df, miss_df, on='gene', how='outer')
	both_df = both_df.fillna(0)

	complete_df = pd.merge(joined, both_df, on='gene', how='left')
	complete_df = complete_df.fillna(0)

	create_plot(complete_df, joined, file_name)


def main():
	input_file = sys.argv[1]
	id_file = sys.argv[2]
	pli_file = sys.argv[3]
	sfari_gene_scores = sys.argv[4]
	mutation_file = sys.argv[5]

	sns.set(rc={'figure.figsize': (18.7, 20.27)})

	sfari_scores = pd.read_csv(sfari_gene_scores, sep='\t').drop(['status', 'number-of-reports'], axis=1)
	sfari_scores.columns = ['gene', 'score', 'syndromic']

	lgd_file_name = input_file
	miss_file_name = lgd_file_name.replace("lgd", "miss")

	pli_file = pd.read_csv(pli_file, sep='\t')
	pli_file = pli_file[['gene', 'pLI', 'oe_lof_upper']]

	mutation_df = pd.read_csv(mutation_file, sep="\t")
	mutation_df['LOF'] = np.where(mutation_df['new_score'] == 1, 'LGD', 'missense')

	lgd_df = process_prob_file(input_file, id_file, lgd_file_name, sfari_scores)
	miss_df = process_prob_file(miss_file_name, id_file, miss_file_name, sfari_scores)
	print("Plotting for LGD variation ranking")
	full_processing(lgd_df, pli_file, sfari_scores, mutation_df, lgd_file_name)
	print("\n")
	print("Plotting for missense variation ranking")
	full_processing(miss_df, pli_file, sfari_scores, mutation_df, miss_file_name)


if __name__ == "__main__":
	main()
