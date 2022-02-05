# preprocessing
# retrieve denovo-db samples
wget https://denovo-db.gs.washington.edu/denovo-db.non-ssc-samples.variants.tsv.gz
wget https://denovo-db.gs.washington.edu/denovo-db.ssc-samples.variants.tsv.gz
cat denovo-db.non-ssc-samples.variants.tsv \
<(tail -n+3 denovo-db.ssc-samples.variants.tsv) > denovo-db.all.variants.tsv
# Download PrimateAI_scores_v0.2.tsv.gz from https://basespace.illumina.com/s/yYGFdGih1rXL
python3 get_db_samples.py \
denovo-db.all.variants.tsv \
PrimateAI_scores_v0.2.tsv


# download constraint data
wget https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz
wget http://genic-intolerance.org/data/RVIS_Unpublished_ExACv2_March2017.txt
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg19/phastCons100way/hg19.100way.phastCons.bw
# bedops: https://bedops.readthedocs.io/en/latest/content/reference/file-management/conversion/wig2bed.html
/software/bedops/2.4.11/x86_64-linux-ubuntu14.04/bin/wig2bed < hg19.100way.phastCons.bw > phastcons_hg19_genes.bed 
# hg19_ucsc2symbol: download coordinates of hg19 genes from https://genome.ucsc.edu/cgi-bin/hgTables, 
# Human, hg19, Genes and Gene Predictions, UCSC Genes track, genome, selected fields: chrom, txStart, txEnd, geneSymbol
bedtools intersect -a phastcons_hg19_genes.bed -b hg19_ucsc2symbol -wb | cut -f1,2,3,4,8 | sort | uniq > phastcons_scores_genes
python3 avg_phastcons.py phastcons_scores_genes

# merge constraint and conservation data
python3 fill_features.py \
RVIS_Unpublished_ExACv2_March2017.txt \
avg_phastcons_elem_gene \
gnomad.v2.1.1.lof_metrics.by_gene.txt


# post-processing
# output_string is the output string provided to the early prediction SNN 
# To generate ROC curves following ensemble prediction, the following script can be run: 
python3 avg_ROC_curve.py \
${output_string}_lgd \
${output_string}_ensemble_lgd_roc_concat \
${output_string}_lgd_individualProb \
${output_string}_ensemble_lgd_roc_roc_concat \

# To generate scatter plots of gene rankings versus constraint metrics, the following script can be run:
python3 test_cases.py \
${output_string}_test_cases_exp \
test_case_ids.txt \
lof_metrics.txt \
SFARI-scores_01-13-2021release_03-30-2021.tsv \
de_novo_mutations.txt 

