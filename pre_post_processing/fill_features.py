# fill_features.py
# given gene scores, fill in missing scores with median value

import sys
import pandas as pd
from functools import reduce


def get_median(df, col_name):
	value = df[col_name].median()
	return value


def gnomad_process(gene_scores):
	gene_scores = gene_scores[[
		'gene', 'obs_mis', 'exp_mis', 'possible_mis', 'obs_syn', 'exp_syn', 'possible_syn',
		'obs_lof', 'exp_lof', 'possible_lof', 'pLI', 'n_sites', 'gene_length', 'oe_lof_upper']]

	# the indices of gene duplicates
	gene_scores = gene_scores.drop([
		17659, 13114, 10736, 16610, 4739, 9060, 10025, 14781, 1532, 19336, 6678, 11396, 15467, 19175, 3569,
		7254, 18422, 10459, 18213, 8135, 9444, 9141, 2548, 12754, 8974, 5085, 17663, 19228, 16160, 16564, 7014, 9266,
		11107, 18006, 15255, 19363, 10721, 8975, 17717, 11326, 13115, 18657, 12755, 9059, 18120, 17355, 4802, 18971,
		10738, 17689, 15946, 9142, 16613, 4740, 7474, 5109, 4977, 9299, 1093, 7790, 12613, 19674, 9055, 10473, 7534,
		858, 16896, 15651, 1194, 11635, 9433, 14783, 12571, 19341, 1702, 11421, 16435, 15469, 5944, 18248, 19180,
		13672, 5314, 16572, 18425, 14504, 12800, 15680, 10311, 8612, 11665, 11336, 2673, 15323, 18217, 6195, 17146,
		6242])
	gene_scores = gene_scores[~pd.isna(gene_scores['gene'])]

	return gene_scores


def write_to_file(file, name):
	file.to_csv(name, sep="\t", header=True, index=False)


def main():
	rvis_file = sys.argv[1]  # /share/hormozdiarilab/Data/GeneScores/RVIS_Unpublished_ExACv2_March2017.txt
	phastcons_file = sys.argv[2]  # /share/hormozdiarilab/Data/GeneScores/phastcons/avg_phastcons_elem_gene
	pli_file = sys.argv[3]  # gnomad_lof_metrics_approved

	rvis_file = pd.read_csv(rvis_file, sep='\t')
	rvis_file = rvis_file[['CCDSr20', 'RVIS[pop_maf_0.05%(any)]']]
	rvis_file.columns = ['gene', 'RVIS']
	phastcons_file = pd.read_csv(phastcons_file, sep="\t")  # closer to 1 == more conserved, more negative selection
	phastcons_file.columns = ['gene', 'phastcons']
	gene_scores = pd.read_csv(pli_file, sep='\t')
	gene_scores = gnomad_process(gene_scores)

	# given a metric, fill in those missing genes with the median score
	rvis_med = get_median(rvis_file, 'RVIS')
	phastcons_med = get_median(phastcons_file, 'phastcons')

	final = reduce(
		lambda left, right: pd.merge(
			left, right, on=['gene'], how='outer'), [gene_scores, rvis_file, phastcons_file])

	result = pd.merge(final, gene_scores['gene'], on='gene')
	result['phastcons'] = result['phastcons'].fillna(phastcons_med)
	result['RVIS'] = result['RVIS'].fillna(rvis_med)
	result = result.sort_values(by='gene')

	write_to_file(result, "lof_metrics.txt")


if __name__ == "__main__":
	main()
