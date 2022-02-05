# avg_phastcons.py
# get the average phastcons score per gene

import pandas as pd
import sys


def main():
	input_file = sys.argv[1]
	output_file = sys.argv[2]

	df = pd.read_csv(
		input_file, sep="\t", header=None, names=['chr', 'start', 'stop', 'score', 'gene']).drop_duplicates()

	# get the average score per gene
	df = (df.groupby('gene')['score'].mean())

	df.to_csv("%s" % output_file, sep="\t", header=False)


if __name__ == "__main__":
	main()
