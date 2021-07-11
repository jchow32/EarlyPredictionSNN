# early_prediction_snn.py
# maximizes true positives and minimizes false positives, takes as input information from de novo LGD and missense
# mutation, conservation + constraint. Optimized parameters: lambda of
# custom loss function based on maximum score estimator, l2 regularizer, batch size, neurons in hidden layer

import argparse
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn.metrics import hinge_loss
import random
import keras.backend as K
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import calibration
from sklearn import linear_model
from math import log2


def get_test_ids(merged_file_y):
    testing_controls = merged_file_y[merged_file_y['PrimaryPhenotype'] == 0].sample(frac=0.25)['ids'].unique().tolist()
    testing_cases = merged_file_y[merged_file_y['PrimaryPhenotype'] == 1].sample(frac=0.25)['ids'].unique().tolist()
    testing_ids = testing_controls + testing_cases
    return testing_ids


def prepare_labels(merged_file):
    # get IDs and their associated label. Determine who will be in test set (25%)
    merged_file_y = merged_file[['ids', 'PrimaryPhenotype']].sort_values(by=['ids']).drop_duplicates()
    testing_ids = get_test_ids(merged_file_y)

    merged_file_y.index = merged_file_y['ids']
    merged_file_y = merged_file_y['PrimaryPhenotype']
    return testing_ids, merged_file_y


def pivot_function(input_df):  # this is to create a matrix where each row is an individual, each column gene
    output_df = input_df[['ids', 'gene', 'new_score']].pivot_table(
        index='ids', columns='gene', values='new_score', aggfunc='mean').fillna(0)
    # if a sample has multiple mutations in a gene, then the mean is taken
    return output_df


def get_train_test(merged_file, merged_file_y, testing_ids):  # split up the test and training/validation set
    x_test = merged_file[merged_file.index.isin(testing_ids)].copy()  # it contains 25% of the data
    x_train = merged_file[~merged_file.index.isin(testing_ids)].copy()  # it contains 75% of the data

    y_test = merged_file_y[merged_file_y.index.isin(testing_ids)].copy()  # corresponding labels for the 25%
    y_train = merged_file_y[~merged_file_y.index.isin(testing_ids)].copy()  # corresponding labels for 75%
    return x_test, x_train, y_test, y_train


def get_probability_per_person(model, x_test, y_test, output_name, action, run_id):
    if "RandomForest" in output_name or "svm" in output_name or "logistic" in output_name:
        y_predict = model.predict_proba(x_test)
        y_predict = pd.DataFrame(data=y_predict)
        y_predict = y_predict[1]
    else:
        y_predict = model.predict(x_test, batch_size=(x_test.shape[0]))

    probabilities = pd.concat(
        [pd.DataFrame(x_test.index), pd.DataFrame(y_predict)], axis=1)
    probabilities.columns = ['ids', 'probability']
    probabilities.index = probabilities['ids']
    probabilities = probabilities.drop(['ids'], axis=1)
    probabilities = pd.concat([probabilities, y_test.to_frame()], axis=1).reset_index()
    probabilities.columns = ['ids', 'probability', 'PrimaryPhenotype']
    probabilities['run_id'] = run_id

    if action == "False":  # if action is false, then we're just ranking genes
        probabilities.to_csv("%s_test_cases_exp" % output_name, sep="\t", index=False, header=False, mode='a')
    else:
        save_to_file(
            probabilities, output_name)
    return probabilities


def max_wrapper(value):
    def max_score_estimator(y_true, y_pred):  # minimize this
        true_positive = K.sum(y_true * y_pred)
        neg_y_true = 1 - y_true
        false_positive = K.sum(neg_y_true * y_pred)
        score = true_positive - (value * false_positive)
        return 1 - score

    return max_score_estimator


def create_prediction_model(x_train, lambda_to_use, predict_nodes, value):
    model = Sequential()
    model.add(Dense(predict_nodes, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(lambda_to_use)))
    model.compile(loss=max_wrapper(value), optimizer='adam', metrics=[tf.keras.metrics.AUC(), max_wrapper(value)])
    return model


def save_to_file(
        probabilities, output_name):
    probabilities.to_csv("%s_individualProb" % output_name, index=False, header=False, sep="\t", mode='a')


def second_phase(
        x_train, y_train, x_test, allen_columns, lambda_to_use, output_name, y_test,
        predict_nodes, action, value, batch_size, run_id):
    scaler = MinMaxScaler()

    if len(allen_columns) > 0:
        x_train[allen_columns] = scaler.fit_transform(x_train[allen_columns])
        x_test[allen_columns] = scaler.transform(x_test[allen_columns])

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    model = create_prediction_model(x_train, lambda_to_use, predict_nodes, value)

    es = EarlyStopping(monitor='val_max_score_estimator', mode='min', patience=2)
    history = model.fit(
        x_train, y_train, epochs=1000, verbose=0, class_weight=class_weights, batch_size=batch_size,
        validation_split=0.2, callbacks=[es])

    probabilities = get_probability_per_person(model, x_test, y_test, output_name, action, run_id)
    return probabilities


def fill_shape(df1, df2):
    missing_columns = list(set(df1.columns) - set(df2.columns))
    to_append = pd.DataFrame(0, index=df2.index, columns=missing_columns).sort_index()
    df2 = pd.concat([df2, to_append], axis=1, sort=False)
    df2 = df2.reindex(sorted(df2.columns), axis=1)
    return df2


def make_same_shape(x_train, x_test):
    x_test = fill_shape(x_train, x_test)
    x_train = fill_shape(x_test, x_train)
    return x_train, x_test


def full_dot_gene_score(df, gene_scores, expression_columns):
    expression_columns_set = {*expression_columns}
    if expression_columns_set.issubset(df):
        df = df.drop(expression_columns, axis=1)
    select_cols = df.columns
    gene_scores = gene_scores[gene_scores.index.isin(select_cols)][
        ['pLI', 'oe_lof_upper', 'RVIS', 'phastcons']]

    # pli and LOEUF have NaN at times-- use 0 for pLI and 1 for LOEUF
    gene_scores['pLI'] = gene_scores['pLI'].fillna(0)
    gene_scores['oe_lof_upper'] = gene_scores['oe_lof_upper'].fillna(1)

    # fill in the missing with median
    missing = (list(set(list(df.columns)) - set(gene_scores.index.tolist())))

    if len(missing) > 0:
        median_expression_columns = gene_scores.median(axis=0).to_frame().T
        row_to_append = pd.DataFrame(
            {'gene': missing}).dropna().reset_index(drop=True)
        median_expression_columns = median_expression_columns.append(
            [median_expression_columns] * (row_to_append.shape[0] - 1), ignore_index=True)
        row_to_append = pd.concat([row_to_append, median_expression_columns], axis=1).set_index('gene')
        partial_file = pd.concat([gene_scores, row_to_append], axis=0)
    else:
        partial_file = gene_scores

    result = df.dot(partial_file)
    return result


def create_rank_cases(gene_to_merge):
    num_test_case_list = gene_to_merge.index.drop_duplicates().to_frame().reset_index(drop=True)

    num_test_case_list['PrimaryPhenotype'] = 1
    num_test_case_list['new_score'] = 1
    num_test_case_list = num_test_case_list.reset_index()
    num_test_case_list['index'] = 'art_case_' + num_test_case_list['index'].astype(str)
    num_test_case_list.columns = ['ids', 'gene', 'PrimaryPhenotype', 'new_score']

    art_case = pivot_function(num_test_case_list)

    art_case_y = art_case.index.to_frame()
    art_case_y['PrimaryPhenotype'] = 1
    art_case_y = art_case_y.drop(['ids'], axis=1).iloc[:, 0]
    return art_case, art_case_y


def get_tpr_low_fpr(tmp_model, x_train, y_train, test):
    # measure how well this set of parameters does for this fold.
    y_predict = tmp_model.predict(x_train.iloc[test], batch_size=(x_train.iloc[test].shape[0]))
    probabilities = pd.concat(
        [pd.DataFrame(x_train.iloc[test].index), pd.DataFrame(y_predict)], axis=1)
    probabilities.columns = ['ids', 'probability']
    probabilities.index = probabilities['ids']
    probabilities = probabilities.drop(['ids'], axis=1)
    probabilities = pd.concat([probabilities, y_train.iloc[test].to_frame()], axis=1).reset_index()
    probabilities.columns = ['ids', 'probability', 'PrimaryPhenotype']

    # first, divide probabilities into cases and controls
    prob_cases = probabilities[probabilities['PrimaryPhenotype'] == 1].sort_values(
        by='probability', ascending=False)
    prob_controls = probabilities[probabilities['PrimaryPhenotype'] == 0].sort_values(
        by='probability', ascending=False)

    num_cases = prob_cases.shape[0]
    highest_control_prob = prob_controls['probability'].iloc[0]

    # get the number of cases with probability higher than the highest control, case fraction
    higher_than_control_cases = prob_cases[
                                    prob_cases['probability'] > highest_control_prob].shape[0] / num_cases

    return higher_than_control_cases


def generic_pick_best(
        x_train, y_train, allen_columns, start_time):
    scaler = MinMaxScaler()

    # x_train and y_train themselves are copies, so they can be modified
    folds = StratifiedKFold(n_splits=3, shuffle=True)
    origin_x_train = x_train.copy()
    origin_y_train = y_train.copy()

    # there are 5 things optimized in the complete model
    batch_size_list = [8, 16, 32]  # batch
    neuron_list = [8, 16, 32]  # neuron
    l_list = [70, 80, 90, 100, 110, 120]  # l
    l2_list = [0.01, 0.001, 0.0001, 0.00001, 0.000001]  # l2

    pick_list = ["batch", "neuron", "custom", "l2"]
    optimal_dict = {}

    for train, test in folds.split(origin_x_train, origin_y_train):

        origin_x_train = x_train.copy()
        origin_y_train = y_train.copy()

        # Split the entire data set into 3 equal folds
        class_weights = class_weight.compute_class_weight(
                        'balanced', np.unique(origin_y_train.iloc[train]), origin_y_train.iloc[train])
        if len(allen_columns) > 0:
            origin_x_train.iloc[train][allen_columns] = scaler.fit_transform(origin_x_train.iloc[train][allen_columns])
            origin_x_train.iloc[test][allen_columns] = scaler.transform(origin_x_train.iloc[test][allen_columns])

        for option_type in pick_list:
            if option_type == "batch":
                options_list = batch_size_list
            elif option_type == "neuron":
                options_list = neuron_list
            elif option_type == "custom":
                options_list = l_list
            else:  # option_type == "l2":
                options_list = l2_list

            for value in options_list:
                key_name = "%s_%s" % (option_type, value)
                print("%s: %s" % (key_name, time.time() - start_time))
                if option_type == "neuron":
                    tmp_model = create_prediction_model(origin_x_train.iloc[train], 0.00001, value, 100)
                elif option_type == "custom":
                    tmp_model = create_prediction_model(origin_x_train.iloc[train], 0.00001, 16, value)
                elif option_type == "l2":
                    tmp_model = create_prediction_model(origin_x_train.iloc[train], value, 16, 100)
                else:
                    tmp_model = create_prediction_model(origin_x_train.iloc[train], 0.00001, 16, 100)

                if option_type == "batch":
                    tmp_model.fit(
                                origin_x_train.iloc[train], origin_y_train.iloc[train], epochs=100, verbose=0,
                                class_weight=class_weights, batch_size=value)
                else:
                    tmp_model.fit(
                                origin_x_train.iloc[train], origin_y_train.iloc[train], epochs=100, verbose=0,
                                class_weight=class_weights, batch_size=32)

                higher_than_control_cases = get_tpr_low_fpr(tmp_model, origin_x_train, origin_y_train, test)
                if key_name in optimal_dict:
                    optimal_dict[key_name].append(higher_than_control_cases)
                else:
                    optimal_dict[key_name] = [higher_than_control_cases]

    # now get the average of every list
    for entry in optimal_dict:
        optimal_dict[entry] = np.mean(optimal_dict[entry])

    # for a given parameter that was optimized, find the best one with the highest average now
    # turn the dictionary into a data frame
    optimal_df = pd.DataFrame(optimal_dict.items(), columns=['parameter', 'fpr_0'])
    # basically just want to know who won, so their parameter value can used, don't need the actual value saved.
    best_batch = get_best_param(optimal_df, 'batch')
    best_neuron = get_best_param(optimal_df, 'neuron')
    best_lam = get_best_param(optimal_df, 'custom')
    best_l2 = get_best_param(optimal_df, 'l2')

    print(best_batch, best_neuron, best_lam, best_l2)

    return int(best_batch), int(best_neuron), int(best_lam), float(best_l2)


def get_best_param(df, search_string):
    relevant_df = df[df.parameter.str.contains(search_string)]
    winner_string = relevant_df.loc[relevant_df['fpr_0'].idxmax(), 'parameter']
    winner = winner_string.split("_")[1]
    return winner


def assess_performance(final_predictions, output_name):
    save_to_file(
        final_predictions, output_name)


def mutation_model(
        start_time, use_four_score, merged_file,
        art_case, art_case_y, gene_to_merge_lgd, output_name, run_id, testing_ids, merged_file_y):
    print("Making feature and score matrix: %s" % (time.time() - start_time))

    merged_file = pivot_function(merged_file)
    merged_file_samples = merged_file.index.tolist()
    merged_file_y = merged_file_y[merged_file_y.index.isin(merged_file_samples)]

    allen_columns = []
    expression_columns = []

    print("Incorporating gene score features: %s" % (time.time() - start_time))
    if use_four_score:
        merged_file_scores = full_dot_gene_score(merged_file, gene_to_merge_lgd, expression_columns)
        art_case_scores = full_dot_gene_score(art_case, gene_to_merge_lgd, expression_columns)
        allen_columns = allen_columns + ['pLI', 'oe_lof_upper', 'RVIS', 'phastcons']

        merged_file = pd.concat([merged_file, merged_file_scores], axis=1)  # make wide
        art_case = pd.concat([art_case, art_case_scores], axis=1)

    x_test, x_train, y_test, y_train = get_train_test(merged_file, merged_file_y, testing_ids)

    # you need to make sure that x_train and x_test have the same number of columns
    x_train, x_test = make_same_shape(x_train, x_test)
    x_train, art_case = make_same_shape(x_train, art_case)
    x_test, art_case = make_same_shape(x_test, art_case)

    # make sure there is no modification of existing data frames
    x_train_optimize = x_train.copy()
    x_train_use = x_train.copy()
    x_train_rank = x_train.copy()
    x_train_prediction = x_train.copy()
    x_test_prediction = x_test.copy()

    x_train_rf = x_train.copy()
    x_train_svm = x_train.copy()
    x_train_log = x_train.copy()

    x_test_rf = x_test.copy()
    x_test_svm = x_test.copy()
    x_test_log = x_test.copy()

    test_prob, train_prob = custom_model_opt_prediction(
        start_time, x_train_optimize, y_train, allen_columns, output_name,
        x_train_rank, x_train_use, x_train_prediction, y_test, x_test_prediction,
        art_case, art_case_y, run_id)

    print("Starting baseline prediction: %s" % (time.time() - start_time))
    rf_test_prob, rf_train_prob = baseline_opt_prediction(
        "%s_RandomForest" % output_name, x_train_rf, y_train, allen_columns, start_time, x_test_rf, y_test,
        run_id)
    svm_test_prob, svm_train_prob = baseline_opt_prediction(
        "%s_svm" % output_name, x_train_svm, y_train, allen_columns, start_time, x_test_svm, y_test,
        run_id)
    log_test_prob, log_train_prob = baseline_opt_prediction(
        "%s_logistic" % output_name, x_train_log, y_train, allen_columns, start_time, x_test_log, y_test,
        run_id)

    return test_prob, train_prob, rf_test_prob, svm_test_prob, log_test_prob


def baseline_fill_dictionary(model, x_train_copy, test, y_train, key_name, optimal_dict, output_name):
    y_predict = model.predict_proba(x_train_copy.iloc[test])
    y_predict = pd.DataFrame(data=y_predict)
    y_predict = y_predict[1]

    probabilities = pd.concat(
        [pd.DataFrame(x_train_copy.iloc[test].index), pd.DataFrame(y_predict)], axis=1)
    probabilities.columns = ['ids', 'probability']
    probabilities.index = probabilities['ids']
    probabilities = probabilities.drop(['ids'], axis=1)
    probabilities = pd.concat([probabilities, y_train.iloc[test].to_frame()], axis=1).reset_index()
    probabilities.columns = ['ids', 'probability', 'PrimaryPhenotype']

    # first, divide probabilities into cases and controls
    prob_cases = probabilities[probabilities['PrimaryPhenotype'] == 1].sort_values(
        by='probability', ascending=False)
    prob_controls = probabilities[probabilities['PrimaryPhenotype'] == 0].sort_values(
        by='probability', ascending=False)

    highest_control_prob = prob_controls['probability'].iloc[0]
    num_total = probabilities.shape[0]

    # for Random Forest:
    if "RandomForest" in output_name:  # Gini Impurity, get closer to 0 is better separation (most pure)
        num_top_cases = prob_cases[prob_cases['probability'] > highest_control_prob].shape[0]
        num_not_top_cases = num_total - num_top_cases
        minimize_this = 1 - ((num_top_cases / num_total) ** 2) - ((num_not_top_cases / num_total) ** 2)
    elif "logistic" in output_name:
        # get the labels as a list, and the probability as a list
        labels = probabilities['PrimaryPhenotype'].tolist()
        prob_list = probabilities['probability'].tolist()
        minimize_this = -sum([labels[i] * log2(prob_list[i]) for i in range(len(labels))])
    else:
        labels = probabilities['PrimaryPhenotype'].tolist()
        prob_list = probabilities['probability'].tolist()
        minimize_this = hinge_loss(labels, prob_list)

    if key_name in optimal_dict:
        optimal_dict[key_name].append(minimize_this)
    else:
        optimal_dict[key_name] = [minimize_this]

    return optimal_dict


def pick_best_random_forest(
        x_train1, y_train, allen_columns, start_time, output_name):
    scaler = MinMaxScaler()

    # be careful about copies. do not touch the originals. That is, don't modify origin or x_train1
    x_train_copy = x_train1.copy()

    estimators = [100, 200, 300, 400, 500]  # for n_estimators
    max_depths = [32, 36, 40, 44, 48, 52]  # for the lambda in the custom loss function

    # make a dictionary to hold the summation of the FPR 0.0 values
    optimal_dict = {}
    folds = StratifiedKFold(n_splits=3, shuffle=True)

    # further split the train into a train set and a test set
    for train, test in folds.split(x_train_copy, y_train):
        # make sure that you don't keep transforming values that were already transformed when they were train
        origin_x_train = x_train1.copy()
        origin_y_train = y_train.copy()

        # Split the entire data set into 3 equal folds
        if len(allen_columns) > 0:
            origin_x_train.iloc[train][allen_columns] = scaler.fit_transform(origin_x_train.iloc[train][allen_columns])
            origin_x_train.iloc[test][allen_columns] = scaler.transform(origin_x_train.iloc[test][allen_columns])

        # loop through these varied lambda and return the loss resulting from using these lambda
        for estimator in estimators:
            for depth in max_depths:
                key_name = "%s_%s" % (estimator, depth)
                print("%s: %s" % (key_name, time.time() - start_time))

                model = RandomForestClassifier(n_estimators=estimator, class_weight='balanced', max_depth=depth)
                model.fit(origin_x_train.iloc[train], origin_y_train.iloc[train])

                optimal_dict = baseline_fill_dictionary(model, origin_x_train, test, origin_y_train, key_name,
                                                        optimal_dict, output_name)

    for entry in optimal_dict:
        optimal_dict[entry] = np.mean(optimal_dict[entry])
    # for a given parameter that was optimized, find the best one with the highest average now
    # turn the dictionary into a data frame
    optimal_df = pd.DataFrame(optimal_dict.items(), columns=['parameter', 'fpr_0'])
    # basically just want to know who won, so their parameter value can used, don't need the actual value saved.
    # for baselines, all combinations are tried exhaustively, so you just need to search for the best one
    b_est, b_depth = get_best_param_random_forest(optimal_df)

    return int(b_est), int(b_depth)


def get_best_param_random_forest(df):
    winner_string = df.loc[df['fpr_0'].idxmin(), 'parameter']
    num_est = winner_string.split("_")[0]
    depth = winner_string.split("_")[1]
    return num_est, depth


def baseline_opt_prediction(
        output_name, x_train, y_train, allen_columns, start_time, x_test, y_test, run_id):
    if "RandomForest" in output_name:
        b_est, b_depth = pick_best_random_forest(
            x_train, y_train, allen_columns, start_time, output_name)
        print("Found optimal parameters: %s" % (time.time() - start_time))

        print("Fitting and prediction started: %s" % (time.time() - start_time))
        test_prob = random_forest_classifier(
            x_train, y_train, x_test, y_test, output_name, allen_columns, b_est, b_depth, run_id)
        print("Fitting and prediction started on train: %s" % (time.time() - start_time))
        train_prob = random_forest_classifier(
            x_train, y_train, x_train, y_train, "%s_train" % output_name, allen_columns, b_est, b_depth, run_id)

    else:  # does logistic or SVM
        b_c = pick_best_l2(
            x_train, y_train, allen_columns, start_time, output_name)
        print("Found optimal parameters: %s" % (time.time() - start_time))

        if "svm" in output_name:
            print("Fitting and prediction started: %s" % (time.time() - start_time))
            test_prob = svm_classifier(x_train, y_train, x_test, y_test, output_name, allen_columns, b_c, run_id)
            print("Fitting and prediction started on train: %s" % (time.time() - start_time))
            train_prob = svm_classifier(x_train, y_train, x_train, y_train, "%s_train" % output_name, allen_columns,
                                        b_c, run_id)
        else:  # does logistic
            print("Fitting and prediction started: %s" % (time.time() - start_time))
            test_prob = logistic_regression(x_train, y_train, x_test, y_test, output_name, allen_columns, b_c, run_id)
            print("Fitting and prediction started on train: %s" % (time.time() - start_time))
            train_prob = logistic_regression(x_train, y_train, x_train, y_train, "%s_train" % output_name,
                                             allen_columns, b_c, run_id)

    return test_prob, train_prob


def svm_classifier(x_train, y_train, x_test, y_test, output_name, allen_columns, c_value, run_id):
    scaler = MinMaxScaler()

    if len(allen_columns) > 0:
        x_train[allen_columns] = scaler.fit_transform(x_train[allen_columns])
        x_test[allen_columns] = scaler.transform(x_test[allen_columns])

    svm_model = svm.LinearSVC(class_weight='balanced', max_iter=10000, C=c_value)
    clf = calibration.CalibratedClassifierCV(svm_model)
    clf.fit(x_train, y_train)
    probability = get_probability_per_person(clf, x_test, y_test, output_name, "True", run_id)

    return probability


def logistic_regression(x_train, y_train, x_test, y_test, output_name, allen_columns, c_value, run_id):
    scaler = MinMaxScaler()

    if len(allen_columns) > 0:
        x_train[allen_columns] = scaler.fit_transform(x_train[allen_columns])
        x_test[allen_columns] = scaler.transform(x_test[allen_columns])

    logistic_model = linear_model.LogisticRegression(class_weight='balanced', max_iter=10000, C=c_value)
    logistic_model.fit(x_train, y_train)
    probability = get_probability_per_person(logistic_model, x_test, y_test, output_name, "True", run_id)
    return probability


def random_forest_classifier(
        x_train, y_train, x_test, y_test, output_name, allen_columns, num_est, depth, run_id):
    scaler = MinMaxScaler()

    if len(allen_columns) > 0:
        x_train[allen_columns] = scaler.fit_transform(x_train[allen_columns])
        x_test[allen_columns] = scaler.transform(x_test[allen_columns])

    # pick the best parameters and then use them
    model = RandomForestClassifier(n_estimators=num_est, class_weight='balanced', max_depth=depth)
    model.fit(x_train, y_train)

    probability = get_probability_per_person(model, x_test, y_test, output_name, "True", run_id)
    return probability


def custom_model_opt_prediction(
        start_time, x_train_optimize, y_train, allen_columns, output_name,
        x_train_rank, x_train_use, x_train_prediction, y_test, x_test_prediction,
        art_case, art_case_y, run_id):
    print("Optimizing parameters for custom model: %s" % (time.time() - start_time))
    b_batch, b_neuron, b_custom_l, b_l2 = generic_pick_best(
        x_train_optimize, y_train, allen_columns, start_time)

    # uncomment if you want to keep track of best parameters over multiple iterations
    # best_values = "%s\t%s\t%s\t%s\n" % (b_batch, b_custom_l, b_l2, b_neuron)
    # with open("%s_custom_best_parameters.txt" % output_name, 'a') as best_file:
    #     best_file.write(best_values)

    print("Start prediction phase: %s" % (time.time() - start_time))
    test_prob = second_phase(
        x_train_prediction, y_train, x_test_prediction, allen_columns, b_l2, output_name, y_test, b_neuron,
        "True", b_custom_l, b_batch, run_id)

    print("Start prediction phase on training: %s" % (time.time() - start_time))
    output_name2 = "%s_train" % output_name
    train_prob = second_phase(
        x_train_use, y_train, x_train_use, allen_columns, b_l2, output_name2, y_train, b_neuron,
        "True", b_custom_l, b_batch, run_id)
    print("Finished: %s" % (time.time() - start_time))

    print("Start gene ranking: %s" % (time.time() - start_time))
    null_prob = second_phase(
        x_train_rank, y_train, art_case, allen_columns, b_l2, output_name, art_case_y,
        b_neuron, "False", b_custom_l, b_batch, run_id)
    print("Finished: %s" % (time.time() - start_time))

    return test_prob, train_prob


def pick_best_l2(
        x_train1, y_train, allen_columns, start_time, output_name):
    scaler = MinMaxScaler()

    x_train_copy = x_train1.copy()

    if "svm" in output_name:
        l2_list = [1, 0.01, 0.001, 10]  # for C, which is 1/lambda
    else:  # is logistic
        l2_list = [10000, 1000, 100, 10, 1]

    # make a dictionary to hold the summation of the FPR 0.0 and FPR 0.01 values
    optimal_dict = {}
    folds = StratifiedKFold(n_splits=3, shuffle=True)

    for train, test in folds.split(x_train_copy, y_train):

        # Split the entire data set into 3 equal folds
        # make sure not double transforming
        origin_x_train = x_train1.copy()
        origin_y_train = y_train.copy()

        if len(allen_columns) > 0:
            origin_x_train.iloc[train][allen_columns] = scaler.fit_transform(origin_x_train.iloc[train][allen_columns])
            origin_x_train.iloc[test][allen_columns] = scaler.transform(origin_x_train.iloc[test][allen_columns])

        # loop through these varied lambda and return the loss resulting from using these lambda
        for l2_value in l2_list:
            key_name = "%s_c" % l2_value
            print("%s: %s" % (key_name, time.time() - start_time))

            if "logistic" in output_name:
                model = linear_model.LogisticRegression(class_weight='balanced', max_iter=1000, C=l2_value)
                model.fit(origin_x_train.iloc[train], origin_y_train.iloc[train])
            else:
                model = svm.LinearSVC(class_weight='balanced', max_iter=100000, C=l2_value)
                model = calibration.CalibratedClassifierCV(model)
                model.fit(origin_x_train.iloc[train], origin_y_train.iloc[train])

            optimal_dict = baseline_fill_dictionary(model, origin_x_train, test, origin_y_train, key_name, optimal_dict,
                                                    output_name)

    for entry in optimal_dict:
        optimal_dict[entry] = np.mean(optimal_dict[entry])
    # for a given parameter that was optimized, find the best one with the highest average now
    # turn the dictionary into a data frame
    optimal_df = pd.DataFrame(optimal_dict.items(), columns=['parameter', 'fpr_0'])
    # basically just want to know who won, so their parameter value can used, don't need the actual value saved.
    # for baselines, all combinations are tried exhaustively, so you just need to search for the best one
    b_c = get_best_param_svm_log(optimal_df)

    return float(b_c)


def get_best_param_svm_log(df):
    winner_string = df.loc[df['fpr_0'].idxmin(), 'parameter']
    c = winner_string.split("_")[0]
    return c


def combined_prediction(lgd_prob, miss_prob, output_name):
    lgd_individuals = lgd_prob['ids'].unique().tolist()
    missense_individuals = miss_prob['ids'].unique().tolist()
    common_individuals = list(set(lgd_individuals).intersection(missense_individuals))

    common_lgd = lgd_prob[lgd_prob['ids'].isin(common_individuals)]
    common_miss = miss_prob[miss_prob['ids'].isin(common_individuals)]
    common = pd.merge(common_lgd[['ids', 'probability', 'PrimaryPhenotype']],
                      common_miss[['ids', 'probability']], on='ids', how='left')
    common.columns = ['ids', 'probability', 'PrimaryPhenotype', 'probability_2']

    common['final'] = common[['probability', 'probability_2']].max(axis=1)
    common = common[['ids', 'final', 'PrimaryPhenotype']]
    common.columns = ['ids', 'probability', 'PrimaryPhenotype']

    assess_performance(common, "%s_combo" % output_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutation", "-m", help="Mutation input file")
    parser.add_argument("--gene", "-g", help="Gene score metrics")
    parser.add_argument("--output", "-o", help="Desired output string")
    parser.add_argument("--lgd", "-l", help="Enable gene score features for LGD-specific model", action='store_true')
    parser.add_argument("--missense", "-m", help="Enable gene score features for missense-specific model",
                        action='store_true')
    args = parser.parse_args()

    input_file = args.mutation
    gene_scores_file = args.gene
    output_name = args.output
    use_four_score = args.lgd
    use_four_score_miss = args.missense

    start_time = time.time()
    # you should create a unique_id based off of the time, so that in the aggregated _individualProb file,
    # you can know which individuals belong to the same independent iteration
    run_id = random.random()

    print("Reading missense and LGD mutations")
    merged_file = pd.read_csv(input_file, sep='\t')  # ids PrimaryPhenotype gene new_score

    print("Reading gnomAD information")
    gene_scores = pd.read_csv(gene_scores_file, sep='\t')
    gene_to_merge_lgd = gene_scores.set_index('gene').sort_index()

    print(
        "Gene ranking: samples each containing single LGD, for every gene in human genome: %s" %
        (time.time() - start_time))
    art_case, art_case_y = create_rank_cases(gene_to_merge_lgd)

    merged_file = merged_file.dropna()
    testing_ids, merged_file_y = prepare_labels(merged_file)

    # this is for missense
    print("\nStarting missense prediction: %s" % (time.time() - start_time))
    merged_file_miss = merged_file[merged_file['new_score'] != 1].dropna()
    if not use_four_score:
        merged_file_miss['new_score'] = 1
    miss_prob, miss_train_prob, m_rf_test_prob, m_svm_test_prob, m_log_test_prob = mutation_model(
        start_time, use_four_score_miss, merged_file_miss,
        art_case, art_case_y, gene_to_merge_lgd, "%s_miss" % output_name, run_id, testing_ids, merged_file_y)

    # this is for LGD
    print("\nStarting LGD prediction: %s" % (time.time() - start_time))
    merged_file_lgd = merged_file[merged_file['new_score'] == 1].dropna()
    lgd_prob, lgd_train_prob, rf_test_prob, svm_test_prob, log_test_prob = mutation_model(
        start_time, use_four_score, merged_file_lgd,
        art_case, art_case_y, gene_to_merge_lgd, "%s_lgd" % output_name, run_id, testing_ids, merged_file_y)

    print("\nStarting both LGD and missense prediction: %s" % (time.time() - start_time))
    # You also have to evaluate the probabilities returned separately, because they will follow different distributions
    combined_prediction(lgd_prob, miss_prob, output_name)
    combined_prediction(rf_test_prob, m_rf_test_prob, "%s_RandomForest" % output_name)
    combined_prediction(svm_test_prob, m_svm_test_prob, "%s_svm" % output_name)
    combined_prediction(log_test_prob, m_log_test_prob, "%s_logistic" % output_name)

    print("\nFinished combined prediction: %s" % (time.time() - start_time))


if __name__ == "__main__":
    main()
