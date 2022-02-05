# get_db_samples.py

# get missense and LGD mutations from NDD cases and controls
# cases from SSC + ASC + Michaelson + Hashimoto + Rauch + DDD
# controls from SSC + GoNL + Gulsuner

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def print_information(df):
    lgd_ids = (df[df['new_score'] == 1]['ids']).unique().tolist()  # LGD in df
    miss_ids = (df[df['new_score'] == 0.5]['ids']).unique().tolist()  # LGD in df

    common_ids = list(set(lgd_ids).intersection(miss_ids))
    print("Combo: %s" % len(common_ids))
    print("Only LGD: %s" % (len(list(set(lgd_ids) - set(common_ids)))))
    print("Only miss: %s" % (len(list(set(miss_ids) - set(common_ids)))))


def filter_d_db(read_ddb, read_primate):

    # just get certain types of variation (missense and LGD)
    read_missense = read_ddb[read_ddb['FunctionClass'].str.contains("missense")].drop_duplicates()
    read_missense = pd.merge(read_missense, read_primate, on=['chr', 'pos', 'ref', 'alt'])
    read_missense['new_score'] = read_missense['primateDL_score']
    read_missense = read_missense[['ids', 'PrimaryPhenotype', 'gene', 'new_score', 'pos']].drop_duplicates()
    read_missense['LOF'] = 0

    # get the LGD variants
    read_lgd = read_ddb[read_ddb['FunctionClass'].str.contains("frameshift") |
                        read_ddb['FunctionClass'].str.contains("stop-") |
                        read_ddb['FunctionClass'].str.contains("splice-")].drop_duplicates()
    read_lgd['new_score'] = 1
    read_lgd = read_lgd[['ids', 'PrimaryPhenotype', 'gene', 'new_score', 'pos']].drop_duplicates()
    read_lgd['LOF'] = 1

    read_ddb = pd.concat([read_missense, read_lgd], ignore_index=True)

    cases = read_ddb[read_ddb['PrimaryPhenotype'] == 1]
    controls = read_ddb[read_ddb['PrimaryPhenotype'] == 0]

    # get list of people who have both missense and LGD
    print_information(cases)
    print_information(controls)

    read_ddb.to_csv("de_novo_mutations.txt", sep='\t', index=False)


def read_in_files(d_db, pai_file):
    read_ddb = pd.read_csv(d_db, sep="\t", skiprows=1, dtype={'Chr': object})
    read_ddb = read_ddb[['#SampleID', 'StudyName', 'PrimaryPhenotype', 'Chr', 'Position', 'Variant',
                         'Gene', 'FunctionClass', 'PolyPhen(HDiv)', 'PolyPhen(HVar)', 'SiftScore']]
    read_ddb.columns = ['ids', 'StudyName', 'PrimaryPhenotype', 'chr', 'pos', 'Variant', 'gene', 'FunctionClass',
                        'polyphen_hdiv', 'polyphen_hvar', 'sift']
    read_ddb['ref'], read_ddb['alt'] = read_ddb['Variant'].str.split('>', 1).str
    read_ddb = read_ddb.drop(['Variant'], axis=1)

    read_ddb = read_ddb[(read_ddb.StudyName == "Iossifov") |
                        (read_ddb.StudyName == "Krumm") |
                        (read_ddb.StudyName == "Turner2016") |
                        (read_ddb.StudyName == "Turner_2017") |
                        (read_ddb.StudyName == "Werling_2018") |
                        ((read_ddb.StudyName == "ASD1_2") & (read_ddb.ids.str.contains("p1"))) |
                        ((read_ddb.StudyName == "ASD3") & (read_ddb.ids.str.contains("s1"))) |
                        (read_ddb.StudyName == "Michaelson2012") |
                        (read_ddb.StudyName == "Hashimoto2015") |
                        ((read_ddb.StudyName == "Rauch2012") & (read_ddb.PrimaryPhenotype != "control")) |
                        (read_ddb.StudyName == "DeRubeis2014") |
                        ((read_ddb.StudyName == "Yuen2017") & (read_ddb.PrimaryPhenotype != "control")) |
                        (read_ddb.StudyName == "Yuen2016") |
                        (read_ddb.PrimaryPhenotype == "developmentalDisorder") |
                        ((read_ddb.StudyName == "GONL") & (read_ddb.PrimaryPhenotype == "control")) |
                        ((read_ddb.StudyName == "Gulsuner2013") & (read_ddb.PrimaryPhenotype == "control"))]

    # replace case and control with 0/1
    read_ddb['PrimaryPhenotype'] = read_ddb['PrimaryPhenotype'].replace(
        ['autism', 'developmentalDisorder', 'intellectualDisability', 'control'], [1, 1, 1, 0])  # case=1, cont=0

    # just concerned with non-synonymous (missense and LGD mutation)
    read_ddb = read_ddb[read_ddb['FunctionClass'].str.contains("frameshift") |
                        read_ddb['FunctionClass'].str.contains("stop-") |
                        read_ddb['FunctionClass'].str.contains("splice-") |
                        read_ddb['FunctionClass'].str.contains("missense")].drop_duplicates()

    read_primate = pd.read_csv(pai_file, sep="\t", skiprows=11, header=0)
    read_primate = read_primate[['chr', 'pos', 'ref', 'alt', 'primateDL_score']]
    read_primate['chr'] = read_primate.chr.str.replace('chr', '')

    return read_ddb, read_primate


def main():
    db = sys.argv[1]
    # unscaled_cadd = sys.argv[2]
    primate = sys.argv[2]
    # output_path = sys.argv[4]
    # coordinates = sys.argv[5]  # hg19_ucsc2symbol_condensed

    read_db, read_primate = read_in_files(db, primate)
    filter_d_db(read_db, read_primate)  # this is for logistic regression using agg. PrimateAI scores


if __name__ == "__main__":
    main()
