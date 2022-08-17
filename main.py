
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

import warnings
warnings.filterwarnings("ignore")

def get_datasets(insulin_data_file_path, cgm_data_file_path, date_time_format):
    if date_time_format == '':
        date_time_format = '%m/%d/%Y %H:%M:%S'
    insulin_dataset_full = pd.read_csv(insulin_data_file_path, low_memory = False)
    insulin_data = insulin_dataset_full[['Date', 'Time', 'BWZ Carb Input (grams)', 'BWZ Estimate (U)']]
    cgm_data_set_full = pd.read_csv(cgm_data_file_path, low_memory = False)
    cgm_data = cgm_data_set_full[['Date', 'Time', 'Sensor Glucose (mg/dL)']]
    cgm_data.dropna(inplace = True)
    insulin_data['DateTime'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'], format = date_time_format)
    cgm_data['DateTime'] = pd.to_datetime(cgm_data['Date'] + " " + cgm_data['Time'], format = date_time_format)
    return insulin_data, cgm_data

def get_meal_start_times_with_insulin_bolus(insulin_dataset):
    insulin_data_filtered = insulin_dataset[insulin_dataset['BWZ Carb Input (grams)'].notna() & insulin_dataset['BWZ Carb Input (grams)'] != 0]
    insulin_data_filtered.rename({'DateTime' : 'MealStartDateTime'}, axis = 1, inplace = True)
    insulin_data_with_insulin_bolus = insulin_data_filtered[['MealStartDateTime', 'BWZ Estimate (U)']]
    insulin_data_with_insulin_bolus.sort_values(by = 'MealStartDateTime', inplace = True)

    meal_start_times_with_insulin_bolus = [(x[0], math.ceil(x[1])) for x in insulin_data_with_insulin_bolus.to_numpy()]
    return meal_start_times_with_insulin_bolus


def get_valid_meal_start_times_with_insulin_bolus(meal_start_times_with_insulin_bolus):
    valid_meal_start_times_with_insulin_bolus = []
    for i in range(len(meal_start_times_with_insulin_bolus)):
        timestamp, insulin_bolus = meal_start_times_with_insulin_bolus[i]
        if i > 0:
            previous_timestamp = meal_start_times_with_insulin_bolus[i-1][0]
            if previous_timestamp > timestamp - timedelta(hours = 0.5):
                continue

        if i < len(meal_start_times_with_insulin_bolus) - 1:
            next_timestamp = meal_start_times_with_insulin_bolus[i+1][0]
            if next_timestamp < timestamp + timedelta(hours = 2):
                continue

        valid_meal_start_times_with_insulin_bolus.append((timestamp, insulin_bolus))
    return valid_meal_start_times_with_insulin_bolus

def extract_meal_and_insulin_bolus_data(cgm_dataset, valid_meal_start_times_with_insulin_bolus):
    cgm_dataset_sorted = cgm_dataset.sort_values(by = 'DateTime')
    meal_data = []
    insulin_bolus_data = []
    cgm_at_meal_start_time_data = []
    for meal_time, insulin_bolus in valid_meal_start_times_with_insulin_bolus:
        start_time = meal_time - timedelta(minutes = 30)
        end_time = meal_time + timedelta(hours = 2)
        filtered_data = cgm_dataset[(cgm_dataset['DateTime'] >= start_time) & (cgm_dataset['DateTime'] <= end_time)]
        if len(filtered_data) > 0:
            meal_data.append(list(filtered_data['Sensor Glucose (mg/dL)'].values))
            insulin_bolus_data.append(insulin_bolus)
            cgm_at_meal_start_time_data.append(cgm_dataset_sorted[cgm_dataset_sorted['DateTime'] >= meal_time]['Sensor Glucose (mg/dL)'].iloc[0])
    return meal_data, insulin_bolus_data, cgm_at_meal_start_time_data

def get_min_max_cgm(meal_data):
    meal_data_df = pd.DataFrame(meal_data)
    return min(meal_data_df.min(axis = 1)), max(meal_data_df.max(axis = 1))

def get_bin_list(min_value, value_list, bin_range = 20):
    bin_list = [int((val - min_value)/bin_range) for val in value_list]
    return bin_list

def get_bins_for_max_cgm(overall_min_cgm, overall_max_cgm, meal_data):
    meal_data_df= pd.DataFrame(meal_data)
    max_cgm_list = meal_data_df.max(axis = 1).values.tolist()
    max_cgm_bin_data = get_bin_list(overall_min_cgm, max_cgm_list, bin_range = 20)
    return max_cgm_bin_data


def get_itemsets(insulin_data_file_path, cgm_data_file_path, date_time_format = '%m/%d/%Y %H:%M:%S'):
    insulin_dataset, cgm_dataset = get_datasets(insulin_data_file_path, cgm_data_file_path, date_time_format)
    meal_start_times_with_insulin_bolus = get_meal_start_times_with_insulin_bolus(insulin_dataset)
    valid_meal_start_times_with_insulin_bolus = get_valid_meal_start_times_with_insulin_bolus(meal_start_times_with_insulin_bolus)
    meal_data, insulin_bolus_data, cgm_at_meal_start_time_data = extract_meal_and_insulin_bolus_data(cgm_dataset, valid_meal_start_times_with_insulin_bolus)
    overall_min_cgm, overall_max_cgm = get_min_max_cgm(meal_data)
    B_max_data = get_bins_for_max_cgm(overall_min_cgm, overall_max_cgm, meal_data)
    B_meal_data = get_bin_list(overall_min_cgm, cgm_at_meal_start_time_data, bin_range = 20)
    
    itemsets = []
    for i in range(len(B_max_data)):
        itemsets.append((B_max_data[i], B_meal_data[i], insulin_bolus_data[i]))
    return itemsets


def retrieve_most_frequent_itemsets(itemsets, support_count_threshold):
    count_dict = {}
    for itemset in itemsets:
        if itemset not in count_dict:
            count_dict[itemset] = 1
        else:
            count_dict[itemset] += 1
        count_dict[itemset]
    itemset_counts = list(count_dict.items())
    itemset_counts.sort(key = lambda x: x[1], reverse = True)
    #print(itemset_counts[0])
    frequent_item_sets_with_count = list(filter(lambda x: x[1] >= support_count_threshold,  itemset_counts))
    frequent_item_sets = [item[0] for item in frequent_item_sets_with_count]
    return frequent_item_sets, frequent_item_sets_with_count, count_dict


#Rules are of the form {Bmax, Bmeal} -> Insulin Bolus 
def get_confidence_of_rules(itemsets, count_dict): 
    precedent_count_dict = {}
    for itemset in itemsets:
        precedent = (itemset[0], itemset[1])
        if precedent not in precedent_count_dict:
            precedent_count_dict[precedent] = 1
        else:
            precedent_count_dict[precedent] += 1

    confidence_dict = {}
    for itemset in itemsets:
        precedent = (itemset[0], itemset[1])
        conf = itemset_count_dict[itemset] / precedent_count_dict[precedent]
        confidence_dict[itemset] = conf

    confidence_list = list(confidence_dict.items())
    return confidence_list


insulin_data_file_path = 'InsulinData.csv'
cgm_data_file_path = 'CGMData.csv'
date_time_format = '%m/%d/%Y %H:%M:%S'

support_count_threshold = 4
low_confidence_threshold = 0.15


itemsets = get_itemsets(insulin_data_file_path, cgm_data_file_path, date_time_format)
frequent_item_sets, frequent_item_sets_with_count, itemset_count_dict = retrieve_most_frequent_itemsets(itemsets, support_count_threshold)
confidence_list = get_confidence_of_rules(itemsets, itemset_count_dict)
confidence_list.sort(key = lambda x: x[1], reverse = True)

highest_confidence = confidence_list[0][1]
highest_confidence_list = list(filter(lambda x: x[1] >= highest_confidence, confidence_list))
low_confidence_list = list(filter(lambda x: x[1] < low_confidence_threshold, confidence_list))


def get_rules(conf_list):
    rules = []
    for rule_with_conf in conf_list:
        itemset = rule_with_conf[0]
        #rule = f'{{{itemset[0]}, {itemset[1]}}} -> {itemset[2]}'
        rule = '{{{0}, {1}}} -> {2}'.format(itemset[0], itemset[1], itemset[2])
        rules.append(rule)
    return rules

highest_confidence_rules = get_rules(highest_confidence_list)
low_confidence_rules = get_rules(low_confidence_list)


frequent_item_sets_str = [str(item) for item in frequent_item_sets]
freq_itemset_df = pd.DataFrame(frequent_item_sets_str)
freq_itemset_df.to_csv('Results_1.csv', index = False, header = False)

highest_conf_rules_df = pd.DataFrame(highest_confidence_rules)
highest_conf_rules_df.to_csv('Results_2.csv', index = False, header = False)


low_conf_rules_df = pd.DataFrame(low_confidence_rules)
low_conf_rules_df.to_csv('Results_3.csv', index = False, header = False)




