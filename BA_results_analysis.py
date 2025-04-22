import json
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import torch


from BA_grid_TASC import determine_outliers_in_grid
from BA_main_experiment import datatype_names
from BA_experiment_resources import attribute_configs_3D, attribute_configs_6D, attribute_values_3D, attribute_values_6D

conf_ids_to_skip = [34, 38, 42, 46]

def determine_index(grid_points, point):
    '''
        Determine index of point within a grid.
    '''
    assert len(point)==len(grid_points)
    index = []
    for i in range(len(point)):
        index.append(grid_points[i].index(point[i]))
    return tuple(index)

def get_corresponding_configs_for_two_fixed_attributes(dimension, s_rank=0, number_of_data_points=0, datatype=0):
    '''
        Returns a dictionary where the values of the variable attribute are the keys and the values are the corresponding configs.
        If dimension is 3 (instead of 6) s_rank can only take the values 1 and 2.
    '''
    # check that exactly two attributes are non-zero
    attribute_names = ["datatype", "number_of_data_points", "s_rank"]
    attributes = [datatype, number_of_data_points, s_rank]
    non_zero_attributes = [i for i,e in enumerate(attributes) if e != 0]
    assert len(non_zero_attributes) == 2
    attribute1_index = non_zero_attributes[0]
    attribute2_index = non_zero_attributes[1]
    temp_list = [i for i,e in enumerate(attributes) if e == 0]
    variable_attribute_index = temp_list[0]
    variable_attribute = attribute_names[variable_attribute_index]
    
    if dimension==3:
        attribute_values = attribute_values_3D
        attribute_configs = attribute_configs_3D
    else:
        attribute_values = attribute_values_6D
        attribute_configs = attribute_configs_6D
    # prepare config_dictionary
    config_dict = dict.fromkeys(attribute_values[variable_attribute])
    for conf_id in attribute_configs.keys():
        attribute2 = attribute_configs[conf_id][attribute2_index]
        attribute1 = attribute_configs[conf_id][attribute1_index]
        # if the non-zero attribute values are the same as the values of this configuration, add this conf_id to the resulting dictionary
        if attributes[attribute1_index] == attribute1 and attributes[attribute2_index] == attribute2:
            key_value = attribute_configs[conf_id][variable_attribute_index]
            config_dict[key_value] = conf_id
    return config_dict


def get_list_of_corresponding_configs(dimension, attribute, attribute2=""):
    ''' 
        Returns a dictionary with all potential values for the attibute as keys and a list of config IDs as values.
        attribute can only take the values "datatype", "number_of_data_points", "s_rank".
        If dimension is 3 (instead of 6) s_rank can only take the values 1 and 2.

        Args:
            attribute (String): configuration attribute of cost function ("datatype", "number_of_data_points", "s_rank")
            attribute2 (String): configuration attribute of cost function ("datatype", "number_of_data_points", "s_rank")
            dimension (int): dimension of cost function (3 or 6)
        Returns:
            dict: Keys are values of attributes, values are lists of config_ids (int)
    '''
    order_attributes = {"datatype":0, "number_of_data_points":1, "s_rank":2, "":-1}
    assert dimension==3 or dimension==6
    assert attribute in ["datatype","number_of_data_points","s_rank"]
    assert attribute2 in order_attributes.keys()
    index = order_attributes[attribute]
    index2 = order_attributes[attribute2]
    if dimension==3:
        attribute_values = attribute_values_3D
        attribute_configs = attribute_configs_3D
    else:
        attribute_values = attribute_values_6D
        attribute_configs = attribute_configs_6D
    config_dict = dict.fromkeys(attribute_values[attribute])
    if index2 >= 0:
        for value in config_dict.keys():
            nested_dict = dict.fromkeys(attribute_values[attribute2])
            for value2 in nested_dict.keys():
                conf_list = []
                for conf_id in attribute_configs.keys():
                    attribute_combination = attribute_configs[conf_id]
                    if(attribute_combination[index]==value and attribute_combination[index2]==value2):
                        conf_list.append(conf_id)
                nested_dict[value2] = conf_list
            config_dict[value] = nested_dict
    else:
        for value in config_dict.keys():
            conf_list = []
            for conf_id in attribute_configs.keys():
                attribute_combination = attribute_configs[conf_id]
                if(attribute_combination[index]==value):
                    conf_list.append(conf_id)
            config_dict[value] = conf_list
    return config_dict

def get_boxplot_standard_deviation_per_attribute(dimension, directory, attribute, value_label):
    '''
        Info about variance of {value_label} values per value of an {attribute}.
        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): directory of json files
            attribute (String): "datatype", "number_of_data_points" or "s_rank". Attribute for x-axis of boxplot
            value_label (String): "TASC", "TSC", "MASC", "MSC". Values for Boxplot
    '''
    assert dimension in [3,6]
    assert attribute in ["datatype", "number_of_data_points", "s_rank"]
    assert value_label in ["TASC", "TSC", "MASC", "MSC"]
    config_lists = get_list_of_corresponding_configs(dimension,attribute)
    if dimension == 3:
        conf_id_list = range(0,32)
    else:
        conf_id_list = range(0,64)
    all_data = load_json_data(directory, conf_id_list=conf_id_list)
    data_dict = {}
    attribute_values = config_lists.keys()
    value_label_dict = "std"
    for value in attribute_values:
        data_dict[value] = []
        for config in config_lists[value]:
            for run in range(len(all_data[config])):
                std = np.asarray(all_data[config][run][value_label][value_label_dict])
                data_dict[value].append(std)
    fig, ax = plt.subplots()
    ax.boxplot(data_dict.values())
    ax.set_xticklabels(data_dict.keys())
    ax.set_title(f"{dimension}D cost function: Standard Deviation of {value_label} per {attribute}")
    ax.set_xlabel(attribute)
    ax.set_ylabel(f"std of {value_label}")
    file_name = f"{dimension}D_{value_label}_std_boxplot_per_{attribute}"
    plt.savefig(f"plots/final_experiment/boxplots/{dimension}D_cost/std/{file_name}.png", dpi=500)
    del all_data

def get_boxplot_fourier_density_per_attribute(dimension, directory, attribute):
    '''
        Info Fourier Density of cost landscape. Simply for confirmation of previous results and own experiment setup.
        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): directory of json files
            attribute (String): "datatype", "number_of_data_points" or "s_rank". Attribute for x-axis of boxplot
            value_label (String): "TASC", "TSC", "MASC", "MSC". Values for Boxplot
    '''
    assert dimension in [3,6]
    assert attribute in ["number_of_data_points", "s_rank"]
    config_lists = get_list_of_corresponding_configs(dimension,"datatype",attribute)
    if dimension == 3:
        conf_id_list = range(0,32)
    else:
        conf_id_list = range(0,64)
    all_data = load_json_data(directory, conf_id_list=conf_id_list)
    data_dict = {}
    
    for datatype in config_lists.keys():
        data_dict[datatype] = {}
        datatype_config_lists = config_lists[datatype]
        attribute_values = list(datatype_config_lists.keys())
        for value in attribute_values:
            data_dict[datatype][value] = []
            for config in datatype_config_lists[value]:
                for run in range(len(all_data[config])):
                    fd = all_data[config][run]["fourier density"]
                    data_dict[datatype][value].append(fd)
    fig, axs = plt.subplots(2,2, figsize=(10,12))
    title_list = list(data_dict.keys())
    for i in range(2):
        for j in range(2):
            index = 2*i + j
            title = title_list[index]
            current_data_dict = data_dict[title]
            axs[i,j].boxplot(current_data_dict.values())
            axs[i,j].set_xticklabels(current_data_dict.keys())
            axs[i,j].set_title(title)
            axs[i,j].set_xlabel(attribute)
            axs[i,j].set_ylabel(f"Fourier Density")

    file_name = f"{dimension}D_fourier_density_boxplot_per_{attribute}"
    plt.tight_layout()
    plt.savefig(f"plots/final_experiment/boxplots/{dimension}D_cost/fd/{file_name}.png", dpi=500)  
    del all_data

def get_outliers_per_config(directory, dimension):
    '''
        Print all outliers found during experiment, which resulting json files are saved in {directory}
    '''
    if dimension == 6:
        no_runs = 5
        no_configs = 64
    else:
        no_runs = 10
        no_configs = 32
    all_data = load_json_data(directory, conf_id_list=range(0,no_configs))
    outlier_dict = {}
    for config in range(no_configs):
        outlier_points = []
        for run in range(no_runs):
            data = all_data[config][run]
            #for metric in ["TASC", "TSC", "MASC", "MSC"]:
            for metric in ["TASC"]:
                tasc_outlier = data[metric]["outlier points"]
                for point in tasc_outlier:
                    if point not in outlier_points:
                        outlier_points.append(point)
        if len(outlier_points)>0:
            print(f"Config {config}: {metric} Outliers at points {outlier_points}")
            outlier_dict[config] = outlier_points
    return outlier_dict

def get_outliers_per_run(directory):
    '''
        Print all outliers found during experiment, which resulting json files are saved in {directory}
    '''
    all_data = load_json_data(directory, conf_id_list=range(0,32))
    outlier_dict = {}
    for config in range(32):
        outlier_dict[config] = {}
        for run in range(10):
            data = all_data[config][run]
            #for metric in ["TASC", "TSC", "MASC", "MSC"]:
            for metric in ["TASC"]:
                tasc_outlier = data[metric]["outlier points"]
                if len(tasc_outlier)>0:
                    outlier_dict[config][run] = tasc_outlier
                    print(f"Config {config} Run {run}: {metric} Outliers at points {tasc_outlier}")
    return outlier_dict


def load_json_data(directory, conf_id_list=range(0,64), only_first_run=False):
    '''
        Load JSON-data for each config_id in conf_id_list from files saved in directory.
        File names start with "conf_{config_id}_" and end with ".json".
        Default conf_id_list is all configs, i.e. 0 to 319.

        Arguments:
            directory (String): source directory for JSON files
            conf_id_list (list of int, optional): list of config_ids 

        Returns:
            all_data (dict): dict where keys are ids and values are a list of all corresponding json-files loaded as dictionaries.
    '''
    all_data = {}
    for id in conf_id_list:
        all_data[id] = []
        for filename in os.listdir(directory):
            prefix = f'config_{id}_'
            if only_first_run:
                prefix = f'config_{id}_run_1'
            if filename.endswith('.json') and prefix in filename:
                file_path = os.path.join(directory, filename)
                #print(f"Lade Datei: {file_path}")
                with open(file_path, 'r') as file:
                    try:
                        all_data[id].append(json.load(file))
                    except json.JSONDecodeError:
                        print(f"Fehler beim Laden der Datei: {file_path}")
        if not all_data:
            print("Keine JSON-Dateien gefunden oder alle Dateien sind fehlerhaft.")
    return all_data

def get_boxplot_per_attribute_fixedDatatype(dimension, directory, attribute, value_label, no_outliers=False):
    '''
        Four Boxplots in one, one plot per data type.
    '''
    assert attribute in ["number_of_data_points", "s_rank"]
    config_dict = get_list_of_corresponding_configs(dimension, "datatype", attribute2=attribute)
    data_dict = dict.fromkeys(config_dict.keys())
    all_data = load_json_data(directory, conf_id_list=range(64))
    print("Attribute:", attribute)
    for datatype in config_dict.keys():
        data_dict[datatype] = {}
        attribute_values = config_dict[datatype].keys()
        value_label_dict = f"{value_label} landscape".lower()
        for value in attribute_values:
            data_dict[datatype][value] = []
            for config in config_dict[datatype][value]:
                for run in range(len(all_data[config])):
                    data_values = np.asarray(all_data[config][run][value_label][value_label_dict])
                    data_values = data_values.flatten().tolist()
                    data_dict[datatype][value].extend(data_values)
        # print summary of boxplots (concrete values)
        print("Datatype:", datatype)
        means, medians, stds = [], [], []
        for value in data_dict[datatype].keys():
            means.append(np.round(np.mean(data_dict[datatype][value]),3))
            medians.append(np.round(np.median(data_dict[datatype][value]),3))
            stds.append(np.round(np.std(data_dict[datatype][value]),3))

        values = list(data_dict[datatype].keys())
        print(f"{attribute} = {values[0]}:     Mean = {means[0]}, Median = {medians[0]}, STD = {stds[0]}") 
        for i in range(1,len(means)):
            mean_diff = np.round(-(means[i-1]-means[i])/means[i-1]*100,1)
            median_diff = np.round(-(medians[i-1]-medians[i])/medians[i-1]*100,1)
            std_diff = np.round(-(stds[i-1]-stds[i])/stds[i-1]*100,1)
            print(f"{attribute} = {values[i]}:     Mean = {means[i]} {mean_diff}%, Median = {medians[i]} {median_diff}%, STD = {stds[i]} {std_diff}%") 
        

    # Plots
    fig, axs = plt.subplots(2,2, figsize=(10,12))
    title_list = list(data_dict.keys())
    for i in range(2):
        for j in range(2):
            index = 2*i + j
            title = title_list[index]
            current_data_dict = data_dict[title]
            if no_outliers:
                axs[i,j].boxplot(current_data_dict.values(), showfliers=False)
                file_name = f"{dimension}D_{value_label}_boxplot_noOutliers_per_{attribute}_fixed_datatype"
            else:
                axs[i,j].boxplot(current_data_dict.values())
                file_name = f"{dimension}D_{value_label}_boxplot_per_{attribute}_fixed_datatype"
            axs[i,j].set_xticklabels(current_data_dict.keys())
            axs[i,j].set_title(title)
            axs[i,j].set_xlabel(attribute)
            axs[i,j].set_ylabel(value_label)

    plt.tight_layout()
    sub_directory = f"plots/final_experiment/boxplots/{dimension}D_cost/fixed_datatype"
    os.makedirs(sub_directory, exist_ok=True)
    plt.savefig(f"{sub_directory}/{file_name}.png", dpi=500)  
    del all_data

def get_boxplot_per_attribute(dimension, directory, attribute, value_label, no_outliers=False):
    '''
        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): directory of json files
            attribute (String): "datatype", "number_of_data_points" or "s_rank". Attribute for x-axis of boxplot
            value_label (String): "TASC", "TSC", "MASC", "MSC". Values for Boxplot
    '''
    assert dimension in [3,6]
    assert attribute in ["datatype", "number_of_data_points", "s_rank"]
    assert value_label in ["TASC", "TSC", "MASC", "MSC"]
    config_lists = get_list_of_corresponding_configs(dimension,attribute)
    if dimension == 3:
        conf_id_list = range(0,32)
    else:
        conf_id_list = range(0,64)
    all_data = load_json_data(directory, conf_id_list=conf_id_list)
    data_dict = {}
    attribute_values = config_lists.keys()
    value_label_dict = f"{value_label} landscape".lower()
    for value in attribute_values:
        data_dict[value] = []
        for config in config_lists[value]:
            for run in range(len(all_data[config])):
                data_values = np.asarray(all_data[config][run][value_label][value_label_dict])
                data_values = data_values.flatten().tolist()
                data_dict[value].extend(data_values)

    # print summary of boxplot (concrete values)
    print("Attribute:", attribute)
    means, medians, stds = [], [], []
    for value in data_dict.keys():
        means.append(np.round(np.mean(data_dict[value]),3))
        medians.append(np.round(np.median(data_dict[value]),3))
        stds.append(np.round(np.std(data_dict[value]),3))

    values = list(data_dict.keys())
    print(f"{attribute} = {values[0]}:     Mean = {means[0]}, Median = {medians[0]}, STD = {stds[0]}") 
    for i in range(1,len(means)):
        mean_diff = np.round((means[i-1]-means[i])/means[i-1]*100,1)
        median_diff = np.round((medians[i-1]-medians[i])/medians[i-1]*100,1)
        std_diff = np.round((stds[i-1]-stds[i])/stds[i-1]*100,1)
        print(f"{attribute} = {values[i]}:     Mean = {means[i]} -{mean_diff}%, Median = {medians[i]} -{median_diff}%, STD = {stds[i]} -{std_diff}%") 
        

    fig, ax = plt.subplots()
    if no_outliers:
        ax.boxplot(data_dict.values(), showfliers=False)
        file_name = f"{dimension}D_{value_label}_boxplot_noOutliers_per_{attribute}"
    else:
        ax.boxplot(data_dict.values())
        file_name = f"{dimension}D_{value_label}_boxplot_per_{attribute}"
    ax.set_xticklabels(data_dict.keys())
    ax.set_title(f"{dimension}D cost function: {value_label} per {attribute}")
    ax.set_xlabel(attribute)
    ax.set_ylabel(value_label)
    
    plt.savefig(f"plots/final_experiment/boxplots/{dimension}D_cost/{file_name}.png", dpi=500)
    del all_data
    
def analyze_fourier_coefficients(dimension, directory, attribute):
    '''
        Determine per configuration or per configuration attribute (datatype, number of data points, Schmidt rank) 
        the distribution of the fourier coefficients.
    '''
    assert dimension in [3,6]
    if dimension==3:
        config_range = range(0,32)
    else:
        config_range = range(0,64)
    all_data = load_json_data(directory, config_range, only_first_run=True)
    conf_lists = get_list_of_corresponding_configs(dimension, attribute)
    data_dict = {}
    attribute_values = conf_lists.keys()
    for value in attribute_values:
        data_dict[value] = []
        for config in conf_lists[value]:
            fcoeff = all_data[config][0]["fourier coefficients"]
            fcoeff,_ = re.subn('\[|\]|\\n', '', fcoeff) 
            fcoeff_norm = np.abs(np.fromstring(fcoeff,dtype=complex,sep=',')).tolist()
            data_dict[value].extend(fcoeff_norm)
    fig, ax = plt.subplots()
    ax.boxplot(data_dict.values())
    ax.set_xticklabels(data_dict.keys())
    ax.set_title(f"{dimension}D cost function: Norm of Fourier Coefficients per {attribute}")
    ax.set_xlabel(attribute)
    ax.set_ylabel(f"Fourier Coefficients")
    file_name = f"{dimension}D_fcoeff_boxplot_per_{attribute}"
    plt.savefig(f"plots/final_experiment/boxplots/{dimension}D_cost/fcoeff/{file_name}.png", dpi=500)
    plt.close()

    n = len(data_dict.keys())
    n1 = 2
    n2 = 2
    if n==2:
        n2 = 1
    fig, axs = plt.subplots(n1, n2, figsize=(10,12))
    title_list = list(data_dict.keys())
    attribute_values = list(data_dict.keys())
    for i in range(n1):
        for j in range(n2):
            if n2 >1:
                ax = axs[i,j]
            else:
                ax = axs[i]
            index = 2*j + i
            value = attribute_values[index]
            title = f"{attribute} = {value}"
            current_fcoeff = data_dict[value]
            counts, bins,_ = ax.hist(current_fcoeff,bins=100,range=(0,0.1))
            ax.set_title(title)
            ax.set_xlabel("Norm Fourier Coefficients")

    file_name = f"{dimension}D_fcoeff_hist_per_{attribute}"
    plt.tight_layout()
    plt.savefig(f"plots/final_experiment/boxplots/{dimension}D_cost/fcoeff/{file_name}.png", dpi=500)  
    plt.close()
    del all_data

def analyze_outlier_point(dimension, directory, point, config_id, run_id):
    '''
        Analyse first and second order gradient at outlier point.
    '''
    all_data = load_json_data(directory, range(config_id, config_id+1))
    run_data = all_data[run_id-1]
    grid_points = run_data["points"]
    point_index = determine_index(grid_points, point)
    shape=[]
    for i in range(dimension):
        shape.append(dimension)
    shape = tuple(shape)
    # Fourier Coefficient
    fcoeff = run_data["fourier coefficients"]
    fcoeff,_ = re.subn('\[|\]|\\n', '', fcoeff) 
    fcoeff_p = np.fromstring(fcoeff,dtype=complex,sep=',').reshape(shape)[point_index]
    print("Fourier Coefficient", fcoeff_p)
    # Gradient 
    grad = run_data["Gradient"]["gradient norm summary landscape"]
    grad_p = grad[point_index]
    print("Gradient summary at outlier", grad_p)
    # Hessian 
    hess = run_data["Hessian"]["hessian norm summary landscape"]
    hess_p = hess[point_index]
    print("Hessian summary at outlier", hess_p)

def check_if_correct_configs():
    directory = "results/main_experiment/6D_cost"
    attribute="datatype"
    for attribute2 in ["s_rank", "number_of_data_points"]:
        att2_keys = {"s_rank": "schmidt Rank", "number_of_data_points": "number of data points"}
        print("datatype", attribute2)
        config_lists = get_list_of_corresponding_configs(6, attribute, attribute2)
        for datatype in config_lists.keys():
            nested_dict = config_lists[datatype]
            for value in nested_dict.keys():
                print("SOLL", datatype, value)
                all_data = load_json_data(directory, nested_dict[value],only_first_run=True)
                for conf in nested_dict[value]:
                    datatypeIST = all_data[conf][0]["data type"]
                    valueIST = all_data[conf][0][att2_keys[attribute2]]
                    print("IST", datatypeIST, valueIST)

def get_boxplot_two_fixed_attributes(dimension, directory, s_rank=0, number_of_data_points=0, datatype=0, metric_name="TASC"):
    '''
        Create boxplot of TASC values where two attributes have fixed values and the third attribute is variable
    '''
    # check that exactly two attributes are non-zero
    attributes = [datatype, number_of_data_points, s_rank]
    non_zero_attributes = [i for i,e in enumerate(attributes) if e != 0]
    assert len(non_zero_attributes) == 2
    # check that label is one of four possible metrics
    assert metric_name in ["TASC", "TSC", "MSC", "MASC"]
    # get corresponding config ids and load experiment result data
    conf_dict = get_corresponding_configs_for_two_fixed_attributes(dimension, s_rank=s_rank, number_of_data_points=number_of_data_points, datatype=datatype)
    all_data = load_json_data(directory=directory, conf_id_list=conf_dict.values())
    # prepare dictionary for boxplot
    boxplot_dict = dict.fromkeys(conf_dict.keys())
    label = metric_name
    landscape_label = f"{label} landscape".lower()
    for value in boxplot_dict.keys():
        data = all_data[conf_dict[value]]
        boxplot_dict[value] = []
        for run in range(len(data)):
            metric_values = data[run][label][landscape_label]
            boxplot_dict[value].extend(np.asarray(metric_values).flatten().tolist())

    # determine title string and variable_attribute depending on fixed attributes
    attribute_names = ["datatype", "number_of_data_points", "s_rank"]
    attribute1_index = non_zero_attributes[0]
    attribute2_index = non_zero_attributes[1]
    attribute1 = attribute_names[attribute1_index]
    attribute2 = attribute_names[attribute2_index]
    attribute1_value = attributes[attribute1_index]
    attribute2_value = attributes[attribute2_index]

    temp_list = [i for i,e in enumerate(attributes) if e == 0]
    variable_attribute_index = temp_list[0]
    variable_attribute = attribute_names[variable_attribute_index]
    
    title_string = f"{attribute1}={attribute1_value}, {attribute2}={attribute2_value}"
    file_name_string = f"{attribute1_value}_{attribute2_value}"

    #y_min = 40000 # minimum value of y axis
    #y_max = 225000 # maximum value of y axis for each variable attribute

    # make boxplot 
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_dict.values())
    ax.set_xticklabels(boxplot_dict.keys())
    ax.set_title(f"{dimension}D cost function: {label} values for\n{title_string}")
    ax.set_xlabel(variable_attribute)
    ax.set_ylabel(label)
    #ax.set_ylim([y_min, y_max])
    file_name = f"{dimension}D_{label}_boxplot_fixed_{file_name_string}.png"
    plot_directory = f"plots/final_experiment/boxplots/{dimension}D_cost/variable_attributes/{label}_values/{variable_attribute}"
    os.makedirs(plot_directory, exist_ok=True)
    plt.savefig(f"{plot_directory}/{file_name}.png", dpi=500)
    plt.close()
    del all_data

def get_median_boxplot_two_fixed_attributes(dimension, directory, s_rank=0, number_of_data_points=0, datatype=0, metric_name="TASC"):
    '''
        Create boxplot of median TASC values for every grid point where two attributes have fixed values and the third attribute is variable.
        Median over each experiment run.
    '''
    # check that exactly two attributes are non-zero
    attributes = [datatype, number_of_data_points, s_rank]
    non_zero_attributes = [i for i,e in enumerate(attributes) if e != 0]
    assert len(non_zero_attributes) == 2
    # check that label is one of four possible metrics
    assert metric_name in ["TASC", "TSC", "MSC", "MASC"]
    # get corresponding config ids and load experiment result data
    conf_dict = get_corresponding_configs_for_two_fixed_attributes(dimension, s_rank=s_rank, number_of_data_points=number_of_data_points, datatype=datatype)
    all_data = load_json_data(directory=directory, conf_id_list=conf_dict.values())
    # prepare dictionary for boxplot
    boxplot_dict = dict.fromkeys(conf_dict.keys())
    label = metric_name
    landscape_label = f"{label} landscape".lower()
    for value in boxplot_dict.keys():
        data = all_data[conf_dict[value]]
        #boxplot_dict[value] = []
        landscapes = []
        for run in range(len(data)):
            metric_values = data[run][label][landscape_label]
            landscapes.append(metric_values)
        boxplot_dict[value] = np.median(landscapes, axis=0).flatten().tolist()

    # determine title string and variable_attribute depending on fixed attributes
    attribute_names = ["datatype", "number_of_data_points", "s_rank"]
    attribute1_index = non_zero_attributes[0]
    attribute2_index = non_zero_attributes[1]
    attribute1 = attribute_names[attribute1_index]
    attribute2 = attribute_names[attribute2_index]
    attribute1_value = attributes[attribute1_index]
    attribute2_value = attributes[attribute2_index]

    temp_list = [i for i,e in enumerate(attributes) if e == 0]
    variable_attribute_index = temp_list[0]
    variable_attribute = attribute_names[variable_attribute_index]
    
    title_string = f"{attribute1}={attribute1_value}, {attribute2}={attribute2_value}"
    file_name_string = f"{attribute1_value}_{attribute2_value}"

    #y_min = 40000 # minimum value of y axis
    #y_max = 225000 # maximum value of y axis for each variable attribute

    # make boxplot 
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_dict.values())
    ax.set_xticklabels(boxplot_dict.keys())
    ax.set_title(f"{dimension}D cost function: median {label} values for\n{title_string}")
    ax.set_xlabel(variable_attribute)
    ax.set_ylabel(label)
    #ax.set_ylim([y_min, y_max])
    file_name = f"{dimension}D_{label}_boxplot_fixed_{file_name_string}.png"
    plot_directory = f"plots/final_experiment/boxplots/{dimension}D_cost/variable_attributes/median_{label}_values/{variable_attribute}"
    os.makedirs(plot_directory, exist_ok=True)
    plt.savefig(f"{plot_directory}/{file_name}.png", dpi=500)
    plt.close()
    del all_data

def make_all_2fixed_boxplots(dimension, directory):
    '''
        Make all boxplots where two training data attributes are fixed.
    '''
    if dimension==3:
        s_rank_range = [1,2]
    elif dimension==6:
        s_rank_range = [1,2,3,4]
    else:
        raise Exception()
    for dt in datatype_names:
        for s in s_rank_range:
            get_boxplot_two_fixed_attributes(dimension, directory, s_rank=s, datatype=dt)
            get_median_boxplot_two_fixed_attributes(dimension, directory, s_rank=s, datatype=dt)
        for ndp in [1,2,3,4]:
            get_boxplot_two_fixed_attributes(dimension, directory, number_of_data_points=ndp, datatype=dt)
            get_median_boxplot_two_fixed_attributes(dimension, directory, number_of_data_points=ndp, datatype=dt)
    for s in s_rank_range:
        for ndp in [1,2,3,4]:
            
            get_boxplot_two_fixed_attributes(dimension, directory, s_rank=s, number_of_data_points=ndp)
            get_median_boxplot_two_fixed_attributes(dimension, directory, s_rank=s, number_of_data_points=ndp)

def get_boxplot_outlier_points_TASC_values(dimension, directory):
    '''
        creates one plot per configuration, that contains the average TASC value for an outlier, as well as 
        one boxplot that includes all other computed average TASC values for points in the landscape.
    '''
    all_data = load_json_data(directory, conf_id_list=range(64))
    average_outlier_dict = {}
    for config in range(64):
        data = all_data[config]
        #create landscape of average TASC values
        landscapes = []
        for run in range(len(data)):
            landscape = data[run]["TASC"]["tasc landscape"]
            landscapes.append(landscape)
        average_landscape = np.mean(landscapes, axis=0)
        points = data[0]["points"]
        outlier_points, outlier_values, outlier_indices = determine_outliers_in_grid(points, average_landscape)
        if len(outlier_points)>0:
            average_outlier_dict[config] = outlier_indices



if __name__=="__main__":
    #get_outliers_per_config("results/main_experiment/6D_cost", dimension=6)
    #outlier_dict = get_outliers_per_run("results/main_experiment/3D_cost")
    directory = "results/main_experiment/6D_cost"
    for a in ["s_rank", "number_of_data_points"]:
        #analyze_fourier_coefficients(6,directory,a)
        #get_boxplot_fourier_density_per_attribute(6, directory, a)
        for metric in ["TASC"]:
            get_boxplot_per_attribute_fixedDatatype(6, directory, a, metric, no_outliers=True)
            get_boxplot_per_attribute_fixedDatatype(6, directory, a, metric, no_outliers=False)
            #get_boxplot_per_attribute(3, directory, a, metric, no_outliers=True)
            #get_boxplot_per_attribute(3, directory, a, metric)
            #get_boxplot_standard_deviation_per_attribute(3, directory, a, metric)
    #make_all_2fixed_boxplots(3, directory)
    #outlier_dict = get_outliers_per_config(directory, 6)
    #get_boxplot_outlier_points_TASC_values(6, directory)