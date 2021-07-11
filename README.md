# EarlyPredictionSNN
## Prediction of neurodevelopmental disorders (NDDs) based on *de novo* coding variation
A shallow neural network (SNN) is able to distinguish NDD cases from controls with high true positive rate (TPR) at very low false positive rates (FPR) compared to traditional machine learning techniques such as random forest, support-vector machine (SVM), and logistic regression, hereon referred to as baseline models.  <br />

### Input parameters <br />

```
--de_novo or -d : Information on *de novo* non-synonymous coding variation for NDD cases and unaffected controls
--gene or -g : Gene-associated constraint and conservation information (gene score information)
--output or -o  : The user provides a desired string ($output_string) used to name output files
--lgd or -l : a flag that, when provided, enables the use of gene score features for the LGD-specific model
--missense or -m : a flag that, when provided, enables the use of gene score features for the missense-specific model
```

For the mutation information (-m), coding *de novo* mutations are retrieved from denovo-db (version 1.6.1), consisting of 6,509 individuals with primary phenotypes of autism spectrum disorder (ASD), intellectual disability, developmental disability, and 1,251 controls. Scores are assigned to coding *de novo* mutation according to the type of non-synonymous mutation present. For likely gene-disruptive (LGD) mutations such as stop, splice, and frameshift mutations, a score of 1 is assigned. For missense mutations, the *PrimateAI* (Sundaram et al. 2018 Nature Genetics) score is used. An example tab-delimited file called *de_novo_mutations.txt* is provided, displaying a sample's ID, the primary phenotype (1 == case, 0 == control), the gene containing a *de novo* mutation, and the score assigned to the mutation. <br />

For the genic constraint and conservation information (-g), pLI, LOEUF, RVIS, and phastCons scores are used describe a gene's relative intolerance to LGD or deleterious mutation and degree of conservation among 99 vertebrates and the human genome. An example tab-delimited file called *lof_metrics.txt* is provided, containing metrics collected by gnomAD (v2.1.1) and RVIS and phastCons values per gene. <br />

For the output file string (-o), If the same $output_string is supplied to multiple runs of the script *early_prediction_snn.py*, predictions are appended to existing predictions; independent iterations can be distinguished from each other by unique run IDs written in output files. Using the provided $output_string, the SNN produces the following output files for LGD- and missense-specific and combined predictions for the SNN and baseline models: <br />
1. ${output_string}\_individualProb , of the tab-delimited format: sample ID<\t>predicted probability of being a case<\t>primary phenotype (1 == case, 0 == control)<\t>unique run ID for this iteration. <br />
2. ${output_string}\test_cases_exp , of the tab-delimited format:  artificial sample ID<\t>predicted probability of being a case<\t>primary phenotype<\t>unique run ID for this iteration. 

The SNN's methods are further described in (URL LINK).

### Example usage

```
python3 early_prediction_snn.py \
-d de_novo_mutations.txt \
-g lof_metrics.txt \
-o $output_string \
-l
```

To generate ensemble predictions, the following script can be run after output files from *early_prediction_snn.py* are generated: 

```
python3 ensemble_prediction.py \
--snn ${output_string}_lgd_individualProb \
--rf ${output_string}_lgd_RandomForest_individualProb \
--svm ${output_string}_lgd_svm_individualProb \
--logistic ${output_string}_lgd_logistic_individualProb \
--output ${output_string}_lgd
```


