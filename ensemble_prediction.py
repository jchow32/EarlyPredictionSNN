# ensemble_prediction.py
# given the predicted probabilities from every model (SNN and baselines), join on common ID and runID
# within an iteration, determine the average (or max) predicted probability for a given individual
# calculate the TPR at FPR < 0.01 for that iteration
# determine the average TPR at FPR < 0.01 across all iterations


import pandas as pd
import numpy as np
from functools import reduce
from sklearn.metrics import roc_curve, precision_recall_curve
import seaborn as sns
from matplotlib import pyplot as plt
import math
from scipy import stats
import argparse


def read_prob_file(in_file, type_string):
    df = pd.read_csv(in_file, sep="\t", header=None)
    prob_string = "%s_prob" % type_string
    df.columns = ['ids', prob_string, 'PrimaryPhenotype', 'run_id']
    return df


def process_prob(df):
    df = df.reset_index()
    df['index'] += 1

    # Find the index of the first occurrence of an identifier.
    all_identifiers = df['run_id'].tolist()
    all_identifiers_unique = df['run_id'].unique().tolist()

    breakpoint_list = []
    for item in all_identifiers_unique:
        id_index = all_identifiers.index(item) + 1
        breakpoint_list.append(id_index)
    breakpoint_list.pop(0)  # so you don't double count the first coordinate

    # create a list of lists that shows the start and end of a set according to 'order'
    coordinate_list = []
    previous_coordinate = 1
    for b_point in breakpoint_list:
        coordinate_list.append([previous_coordinate, b_point - 1])
        previous_coordinate = b_point
    # add the last one which was left out, also get the 'order' of the very last entry from prob_df
    last_order = df['index'].iloc[-1]
    if breakpoint_list:  # there are multiple iterations
        last_coordinate = breakpoint_list[-1]
        coordinate_list.append([last_coordinate, last_order])
    else:  # the list is empty, so there's only 1 iteration
        coordinate_list = [[1, df.shape[0]]]
    return df, coordinate_list


def get_tpr(start_index, end_index, prob_df, type_of_prob, tpr_list):
    # now select from prob_df 'order' a single iteration
    one_set = prob_df.iloc[start_index:end_index].sort_values(by=type_of_prob, ascending=False)
    # one set shows all the associated probabilities from the SNN, RF, SVM, LR for an iteration

    # the number of samples in this iteration
    num_cases = one_set[one_set['PrimaryPhenotype'] == 1].shape[0]
    top_control_prob = one_set[one_set['PrimaryPhenotype'] == 0].iloc[0][type_of_prob]  # for the highest
    # TPR at FPR < 0.01
    # return for each set, and then find the average of all sets
    # retrieve the top cases with probability above top_control_prob
    top_cases = one_set[one_set[type_of_prob] > top_control_prob]
    tpr = top_cases.shape[0] / num_cases
    tpr_list.append(tpr)
    return tpr_list, one_set


def process_coordinate(pair, prob_df, tpr_list, tpr_list_no_snn, type_of_prob, type_of_prob_no_snn, t_list_snn,
                       t_list_rf, t_list_svm, t_list_lr, t_list_rand):
    start_index = pair[0] - 1
    end_index = pair[1]

    # the highest probability associated with a control
    tpr_list, one_set = get_tpr(start_index, end_index, prob_df, type_of_prob, tpr_list)
    tpr_list_no_snn, one_dummy = get_tpr(
        start_index, end_index, prob_df, type_of_prob_no_snn, tpr_list_no_snn)

    # you could also measure the tpr for the other methods, so you can also print here
    t_list_snn, one_set = get_tpr(start_index, end_index, prob_df, 'custom_prob', t_list_snn)
    t_list_rf, one_set = get_tpr(start_index, end_index, prob_df, 'rf_prob', t_list_rf)
    t_list_svm, one_set = get_tpr(start_index, end_index, prob_df, 'svm_prob', t_list_svm)
    t_list_lr, one_set = get_tpr(start_index, end_index, prob_df, 'log_prob', t_list_lr)
    t_list_rand, one_set = get_tpr(start_index, end_index, prob_df, 'random_prob', t_list_rand)

    return tpr_list, tpr_list_no_snn, one_set, t_list_snn, t_list_rf, t_list_svm, t_list_lr, t_list_rand


def find_thresholds(one_set, auc_list, prob_type, out_name, auc_type):
    fpr, tpr, thresholds = roc_curve(one_set['PrimaryPhenotype'], one_set[prob_type])
    fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    interested_rates = np.arange(0, 1.01, 0.01)
    interested_df = pd.DataFrame(columns=['FPR', 'TPR'])
    for rate in interested_rates:
        if auc_type == "roc":
            value_01 = fpr_tpr[fpr_tpr['FPR'] <= rate].tail(
                1)  # for ROC, trying to find the largest TPR at the rate
        else:
            value_01 = fpr_tpr[fpr_tpr['FPR'] >= rate].head(1)  # for PR, that's the largest recall at the rate
            # round FPR down to nearest 0.01, and then group by the recall (FPR) column
        interested_df = interested_df.append(value_01)
    interested_df.index = interested_rates
    if out_name != "blank":
        interested_df.to_csv(out_name, header=False, sep="\t", mode='a')
    auc = np.trapz(interested_df['TPR'], x=interested_df['FPR'])
    auc_list.append(auc)

    return interested_df, auc_list


def get_auc(
        one_set, out_name, c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list, auc_type):

    if auc_type == "roc":  # the ROC AUC per iteration
        c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list = \
            write_roc(one_set, c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list, "%s_roc" % out_name, "ROC")
    else:  # precision recall AUC type
        c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list = \
            write_roc(one_set, c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list, out_name, "PR")

    return c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list


def write_roc(one_set, c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list, out_name, write_type):
    set_prob = one_set[['custom_prob', 'rf_prob', 'svm_prob', 'log_prob', 'avg_prob', 'avg_noSNN_prob',
                        'PrimaryPhenotype']]
    if write_type == "PR":
        c_pr, c_list = get_pr(set_prob, 'custom_prob', c_list)
        rf_pr, rf_list = get_pr(set_prob, 'rf_prob', rf_list)
        svm_pr, svm_list = get_pr(set_prob, 'svm_prob', svm_list)
        log_pr, log_list = get_pr(set_prob, 'log_prob', log_list)
        ens_pr, e_list = get_pr(set_prob, 'avg_prob', e_list)
        ens_no_snn_pr, e_no_snn_list = get_pr(set_prob, 'avg_noSNN_prob', e_no_snn_list)

        merged_df = pd.concat([c_pr, rf_pr, svm_pr, log_pr, ens_pr, ens_no_snn_pr], axis=1)
        merged_df.to_csv("%s_concat" % out_name, sep="\t", header=False, mode='a')
    else:
        # writing ROC concat file
        c_roc, c_list = find_thresholds(set_prob, c_list, 'custom_prob', 'blank', "roc")
        rf_roc, rf_list = find_thresholds(set_prob, rf_list, 'rf_prob', 'blank', "roc")
        svm_roc, svm_list = find_thresholds(set_prob, svm_list, 'svm_prob', 'blank', "roc")
        log_roc, log_list = find_thresholds(set_prob, log_list, 'log_prob', 'blank', "roc")
        e_roc, e_list = find_thresholds(set_prob, e_list, 'avg_prob', 'blank', "roc")
        e_no_snn_roc, e_no_snn_list = find_thresholds(set_prob, e_no_snn_list, 'avg_noSNN_prob', 'blank', "roc")

        merged_df = pd.concat([c_roc, rf_roc, svm_roc, log_roc, e_roc, e_no_snn_roc], axis=1)
        merged_df.to_csv("%s_concat" % out_name, sep="\t", header=False, mode='a')

    return c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list


def get_pr(set_prob, prob_type, auc_list):
    c_p, c_r, c_t = precision_recall_curve(set_prob['PrimaryPhenotype'], set_prob[prob_type])
    fpr_tpr = pd.DataFrame({'recall': c_r, 'precision': c_p})
    # set the threshold column, which is the recall column rounded down
    interested_rates = np.arange(0, 1.01, 0.01)[::-1]
    interested_df = pd.DataFrame(columns=['recall', 'precision'])
    for rate in interested_rates:
        value_01 = fpr_tpr[fpr_tpr['recall'] >= rate].tail(1)  # for PR, that's the largest recall at the rate
        # round FPR down to nearest 0.01, and then group by the recall (FPR) column
        interested_df = interested_df.append(value_01)
    interested_df.index = interested_rates

    # calculate the AUC for the PR curve
    interested_df = interested_df.sort_values(by='recall')
    auc = np.trapz(interested_df['precision'], x=interested_df['recall'])
    # the x-axis has to be sorted in ascending order
    auc_list.append(auc)

    return interested_df, auc_list


def print_tpr_auc(coordinate_list, df_merged, avg_name, avg_no_snn_name, out_name):

    # retrieving TPR at FPR < 0.01 per iteration, getting mean. Must retain values for plotting confidence intervals
    t_list_snn = []
    t_list_rf = []
    t_list_svm = []
    t_list_lr = []
    tpr_list_avg = []
    tpr_list_no_snn_avg = []
    t_list_rand = []

    # retrieving ROC AUC from each iteration
    c_list = []
    rf_list = []
    svm_list = []
    log_list = []
    e_list = []
    e_no_snn_list = []
    r_auc = []

    # retrieving PR-AUC from each iteration
    p_c_list = []
    p_rf_list = []
    p_svm_list = []
    p_log_list = []
    p_e_list = []
    p_e_no_snn_list = []
    r_pr_auc = []

    for pair in coordinate_list:
        tpr_list_avg, tpr_list_no_snn_avg, one_set, t_list_snn, t_list_rf, t_list_svm, t_list_lr, t_list_rand = \
            process_coordinate(pair, df_merged, tpr_list_avg, tpr_list_no_snn_avg, avg_name, avg_no_snn_name,
                               t_list_snn, t_list_rf, t_list_svm, t_list_lr, t_list_rand)

        # for every type of probability, get the precision recall values and append them to file
        c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list = get_auc(
            one_set, out_name, c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list, "roc")
        # do this separately for precision and recall, add an argument to that tells if PR AUC
        p_c_list, p_rf_list, p_svm_list, p_log_list, p_e_list, \
            p_e_no_snn_list = get_auc(
                one_set, out_name, p_c_list, p_rf_list, p_svm_list, p_log_list, p_e_list, p_e_no_snn_list, "pr")

        r_thresh, r_auc = find_thresholds(one_set, r_auc, 'random_prob', 'blank', 'roc')

        num_cases = one_set[one_set['PrimaryPhenotype'] == 1].shape[0]
        num_controls = one_set[one_set['PrimaryPhenotype'] == 0].shape[0]
        random_pr = num_cases / (float(num_cases) + float(num_controls))  # in Python3, it produces a float anyways
        r_pr_auc.append(random_pr)

    print("Using SNN: Avg. ROC AUC: %.5f" % (np.mean(c_list)))
    print("Using RF: Avg. ROC AUC: %.5f" % (np.mean(rf_list)))
    print("Using SVM: Avg. ROC AUC: %.5f" % (np.mean(svm_list)))
    print("Using LR: Avg. ROC AUC: %.5f" % (np.mean(log_list)))
    print("Using avg. ensemble: Avg. ROC AUC: %.5f" % (np.mean(e_list)))
    print("Using avg. ensemble - SNN: Avg. ROC AUC: %.5f" % (np.mean(e_no_snn_list)))
    print("Random ROC AUC: %.5f:" % (np.mean(r_auc)))
    print("\n")

    print("Calculating 95% confidence intervals for ROC AUC:")
    get_confidence_interval(c_list, rf_list, svm_list, log_list, e_list, e_no_snn_list,
                            r_auc)
    snn_z = random_z_score(c_list, r_auc)
    rf_z = random_z_score(rf_list, r_auc)
    svm_z = random_z_score(svm_list, r_auc)
    log_z = random_z_score(log_list, r_auc)
    e_z = random_z_score(e_list, r_auc)
    e_ns_z = random_z_score(e_no_snn_list, r_auc)
    print("Z-score for difference of SNN and Random: %.5f" % snn_z)
    print("Z-score for difference of RF and Random: %.5f" % rf_z)
    print("Z-score for difference of SVM and Random: %.5f" % svm_z)
    print("Z-score for difference of LR and Random: %.5f" % log_z)
    print("Z-score for difference of Ensemble and Random: %.5f" % e_z)
    print("Z-score for difference of Ensemble - SNN and Random: %.5f" % e_ns_z)

    # TPR at FPR < 0.01
    print("Using SNN: Avg. TPR < 0.01: %.5f" % (np.mean(t_list_snn)))
    print("Using RF: Avg. TPR < 0.01: %.5f" % (np.mean(t_list_rf)))
    print("Using SVM: Avg. TPR < 0.01: %.5f" % (np.mean(t_list_svm)))
    print("Using LR: Avg. TPR < 0.01: %.5f" % (np.mean(t_list_lr)))
    print("Using avg. ensemble: Avg. TPR < 0.01: %.5f" % (np.mean(tpr_list_avg)))
    print("Using avg. ensemble - SNN: Avg. TPR < 0.01: %.5f" % (np.mean(tpr_list_no_snn_avg)))
    print("Random TPR at FPR < 0.01: %.5f" % (np.mean(t_list_rand)))
    print("\n")

    print("Calculating 95% confidence intervals of TPR at FPR < 0.01:")
    bar_df = get_confidence_interval(t_list_snn, t_list_rf, t_list_svm, t_list_lr, tpr_list_avg, tpr_list_no_snn_avg,
                                     t_list_rand)

    snn_z = random_z_score(t_list_snn, t_list_rand)
    rf_z = random_z_score(t_list_rf, t_list_rand)
    svm_z = random_z_score(t_list_svm, t_list_rand)
    log_z = random_z_score(t_list_lr, t_list_rand)
    e_z = random_z_score(tpr_list_avg, t_list_rand)
    e_ns_z = random_z_score(tpr_list_no_snn_avg, t_list_rand)
    print("TPR at FPR < 0.01:")
    print("Z-score for difference of SNN and Random: %.5f" % snn_z)
    print("Z-score for difference of RF and Random: %.5f" % rf_z)
    print("Z-score for difference of SVM and Random: %.5f" % svm_z)
    print("Z-score for difference of LR and Random: %.5f" % log_z)
    print("Z-score for difference of Ensemble and Random: %.5f" % e_z)
    print("Z-score for difference of Ensemble - SNN and Random: %.5f" % e_ns_z)

    print("\n")

    # the RF curves AUC
    print("Using SNN: Avg. PR AUC: %.5f" % (np.mean(p_c_list)))
    print("Using RF: Avg. PR AUC: %.5f" % (np.mean(p_rf_list)))
    print("Using SVM: Avg. PR AUC: %.5f" % (np.mean(p_svm_list)))
    print("Using LR: Avg. PR AUC: %.5f" % (np.mean(p_log_list)))
    print("Using Ensemble: Avg. PR AUC: %.5f" % (np.mean(p_e_list)))
    print("Using Ensemble - SNN: Avg. PR AUC: %.5f" % (np.mean(p_e_no_snn_list)))

    print("Calculating 95% confidence intervals for PR-AUC:")
    get_confidence_interval(p_c_list, p_rf_list, p_svm_list, p_log_list, p_e_list, p_e_no_snn_list,
                            r_pr_auc)
    print("\n")

    # TODO Calculate the z-score of taking the model's PR-AUC, subtracting random PR-AUC from it
    # TODO what's happening with this missense
    # something = pd.DataFrame(p_c_list)
    # something.to_csv("CheckMissenseSNN", sep="\t", header=None)
    snn_z = random_z_score(p_c_list, r_pr_auc)
    rf_z = random_z_score(p_rf_list, r_pr_auc)
    svm_z = random_z_score(p_svm_list, r_pr_auc)
    log_z = random_z_score(p_log_list, r_pr_auc)
    e_z = random_z_score(p_e_list, r_pr_auc)
    e_ns_z = random_z_score(p_e_no_snn_list, r_pr_auc)

    print("Z-score for difference of SNN and Random: %.5f" % snn_z)
    print("Z-score for difference of RF and Random: %.5f" % rf_z)
    print("Z-score for difference of SVM and Random: %.5f" % svm_z)
    print("Z-score for difference of LR and Random: %.5f" % log_z)
    print("Z-score for difference of Ensemble and Random: %.5f" % e_z)
    print("Z-score for difference of Ensemble - SNN and Random: %.5f" % e_ns_z)
    print("\n")

    return tpr_list_avg, tpr_list_no_snn_avg, bar_df


def random_z_score(df, rand_pr_auc):
    df = pd.DataFrame(df, columns=['PR_AUC_value'])
    rand_df = pd.DataFrame(rand_pr_auc, columns=['PR_AUC_random'])
    df_merge = pd.concat([df, rand_df], axis=1)

    # # subtract random from non-random and assign to new column
    # df_merge['diff'] = df_merge['PR_AUC_value'] - df_merge['PR_AUC_random']
    # mean_diff = np.mean(df_merge['diff'])
    # std_diff = np.std(df_merge['diff'])
    # df_merge['z_score'] = (df_merge['diff'] - mean_diff) / std_diff
    #
    # # now get the mean of the z-scores for that model
    # mean_z_score = np.mean(df_merge['z_score'])

    # TODO some other way of doing it
    mean_pr = np.mean(df_merge['PR_AUC_value'])
    mean_rand_pr = np.mean(df_merge['PR_AUC_random'])
    var_pr = np.var(df_merge['PR_AUC_value'])
    var_rand_pr = np.var(df_merge['PR_AUC_random'])

    print("Variance of model: %.5f" % var_pr)

    z_score = (mean_pr - mean_rand_pr) / (np.sqrt((var_pr + var_rand_pr)))

    return z_score


def processing(custom_file, rf_file, svm_file, log_file, out_name):

    custom_df = read_prob_file(custom_file, 'custom')
    svm_df = read_prob_file(svm_file, 'svm').drop(['PrimaryPhenotype'], axis=1)
    log_df = read_prob_file(log_file, 'log').drop(['PrimaryPhenotype'], axis=1)
    rf_df = read_prob_file(rf_file, 'rf').drop(['PrimaryPhenotype'], axis=1)

    # make a random one too
    random_df = custom_df.drop(['custom_prob', 'PrimaryPhenotype'], axis=1)
    random_df['random_prob'] = np.random.uniform(0, 1, random_df.shape[0])

    df_list = [custom_df, rf_df, svm_df, log_df, random_df]
    tpr_list, tpr_list_no_snn, df_merged, bar_df = get_tpr_list(df_list, out_name)

    return df_merged, tpr_list, tpr_list_no_snn, bar_df


def drop_other_prob(df):
    df = df.drop(['custom_prob', 'rf_prob', 'svm_prob', 'log_prob'], axis=1)
    return df


def get_tpr_list(df_list, out_name):
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['ids', 'run_id']), df_list)
    df_merged, coordinate_list = process_prob(df_merged)

    df_merged['avg_noSNN_prob'] = df_merged[['rf_prob', 'svm_prob', 'log_prob']].mean(axis=1)
    df_merged['avg_prob'] = df_merged[['custom_prob', 'rf_prob', 'svm_prob', 'log_prob']].mean(axis=1)

    tpr_list, tpr_list_no_snn, bar_df = print_tpr_auc(coordinate_list, df_merged, 'avg_prob', 'avg_noSNN_prob',
                                                      out_name)
    return tpr_list, tpr_list_no_snn, df_merged, bar_df


def determine_max(df, col):
    col1 = "%s_y" % col
    col2 = "%s_x" % col
    df[col] = df[[col1, col2]].max(axis=1)
    df = df.drop([col1, col2], axis=1)
    return df


def get_max_prob_combo(df):
    df = determine_max(df, 'custom_prob')
    df = determine_max(df, 'rf_prob')
    df = determine_max(df, 'svm_prob')
    df = determine_max(df, 'log_prob')
    df = determine_max(df, 'avg_prob')
    df = determine_max(df, 'avg_noSNN_prob')
    return df


def combined_prediction(lgd_prob, miss_prob, out_name):
    lgd_individuals = lgd_prob['ids'].unique().tolist()
    missense_individuals = miss_prob['ids'].unique().tolist()
    common_individuals = list(set(lgd_individuals).intersection(missense_individuals))

    common_lgd = lgd_prob[lgd_prob['ids'].isin(common_individuals)].drop(['index'], axis=1)
    common_miss = miss_prob[
        miss_prob['ids'].isin(common_individuals)].drop(['index'], axis=1).drop(['PrimaryPhenotype'], axis=1)

    # join on the run_id and the ids
    df_list = [common_lgd, common_miss]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['ids', 'run_id']), df_list)

    # from here you should determine the true combo value to return for SNN, RF, SVM, LR
    df_merged = get_max_prob_combo(df_merged)

    # now loop through each iteration and calculate the TPR at FPR < 0.01
    # first add the index
    df_merged, coordinate_list = process_prob(df_merged)
    df_merged['random_prob'] = np.random.uniform(0, 1, df_merged.shape[0])
    tpr_list, tpr_list_no_snn, bar_df = print_tpr_auc(coordinate_list, df_merged, 'avg_prob', 'avg_noSNN_prob',
                                                      out_name)

    return tpr_list, tpr_list_no_snn, bar_df


def list_to_df(input_list, model_name):
    df = pd.DataFrame({'TPR': input_list})
    df['Model'] = model_name
    return df


def add_model_label(df, model_name):
    df['Model'] = model_name
    return df


def get_confidence_interval(t_list_snn, t_list_rf, t_list_svm, t_list_lr, tpr_list_avg, tpr_list_no_snn_avg,
                            t_list_rand):
    # first convert the lists into df with model labels
    snn = list_to_df(t_list_snn, 'SNN')
    rf = list_to_df(t_list_rf, "Random Forest")
    svm = list_to_df(t_list_svm, "SVM")
    logistic = list_to_df(t_list_lr, "Logistic Regression")
    ensemble = list_to_df(tpr_list_avg, "Ensemble")
    ensemble_ns = list_to_df(tpr_list_no_snn_avg, "Ensemble - SNN")
    rand = list_to_df(t_list_rand, "Random")

    # concatenate all of these data frames in a long way
    merge_df_all = pd.concat([snn, rf, svm, logistic, ensemble, ensemble_ns, rand], axis=0)
    merge_df = pd.concat([snn, rf, svm, logistic, ensemble, ensemble_ns], axis=0)

    # you shouldn't plot random, but you should calculate the 95% CI for random
    fig = plt.figure()
    bar_1 = sns.barplot(x='Model', y='TPR', data=merge_df_all)

    order_list = [
        'SNN', 'Random Forest', 'SVM', "Logistic Regression", "Ensemble", "Ensemble - SNN", "Random"]
    for p, situation in zip(bar_1.lines, order_list):
        xy = p.get_xydata()
        lower = round(xy[0][1], 4)
        upper = round(xy[1][1], 4)
        print("%s: 95%% CI: (%s, %s)" % (situation, lower, upper))

    return merge_df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--snn", "-s", help="SNN LGD-specific predicted probabilities")
    parser.add_argument("--rf", "-r", help="Random Forest LGD-specific predicted probabilities")
    parser.add_argument("--svm", "-v", help="SVM LGD-specific predicted probabilities")
    parser.add_argument("--logistic", "-l", help="Logistic regression LGD-specific predicted probabilities")
    parser.add_argument("--output", "-o", help="Output string of the format ${output_string}_lgd")
    args = parser.parse_args()

    custom_file = args.snn
    rf_file = args.rf
    svm_file = args.svm
    log_file = args.logistic
    out_name = args.output

    # these are the missense files
    custom_miss_file = custom_file.replace('lgd', 'miss')
    rf_miss_file = rf_file.replace('lgd', 'miss')
    svm_miss_file = svm_file.replace('lgd', 'miss')
    log_miss_file = log_file.replace('lgd', 'miss')

    stem_string = custom_file.replace('lgd_individualProb', '')
    print("LGD")
    lgd_merged, lgd_tpr_list, lgd_tpr_list_no_snn, lgd_bar_df = processing(
        custom_file, rf_file, svm_file, log_file, "%sensemble_lgd_roc" % stem_string)
    print("Missense")
    miss_merged, miss_tpr_list, miss_tpr_list_no_snn, miss_bar_df = processing(
        custom_miss_file, rf_miss_file, svm_miss_file, log_miss_file, "%sensemble_miss_roc" % stem_string)
    print("Combined")
    combo_tpr_list, combo_tpr_list_no_snn, combo_bar_df = combined_prediction(
        lgd_merged, miss_merged, "%sensemble_combo_roc" % stem_string)  # don't do this anymore

    # plot the bar chart that has 3 kinds of mutations and all models
    # you have to add a 'Mutation' column to each separate df
    lgd_bar_df['Mutation'] = "LGD"
    miss_bar_df['Mutation'] = "Missense"
    combo_bar_df['Mutation'] = "Combined"
    total_bar_df = pd.concat([lgd_bar_df, miss_bar_df, combo_bar_df], axis=0)
    fig = plt.figure()
    bar_2 = sns.barplot(x='Mutation', y='TPR', hue='Model', data=total_bar_df)
    plt.savefig('%s_full_bar.png' % out_name)

    no_ensemble = total_bar_df[(total_bar_df['Model'] != "Ensemble") & (total_bar_df['Model'] != "Ensemble - SNN")]
    fig = plt.figure()
    bar_2 = sns.barplot(x='Mutation', y='TPR', hue='Model', data=no_ensemble)
    plt.savefig('%s_full_bar_noEnsemble.png' % out_name)


if __name__ == "__main__":
    main()
