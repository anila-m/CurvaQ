import os
import re
from matplotlib import pyplot as plt
import numpy as np

from BA_grid_TASC import determine_outliers_in_grid
from BA_main_experiment import datatype_names
from BA_experiment_resources import attribute_names
from BA_analysis_util import get_corresponding_configs_for_two_fixed_attributes, get_list_of_corresponding_configs, load_json_data, make_boxplot, print_relative_changes, outlier_aware_hist,low_color2,high_color2

######### non-Thesis Experiment Results Plots #########
def get_boxplot_two_fixed_attributes(dimension, directory, s_rank=0, number_of_data_points=0, datatype=0, metric_name="TASC", median=False):
    '''
        Create boxplot of metric values where two attributes have fixed values and the third attribute is variable.
        If median is True, the median metric values over each run are used instead of all runs.

        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): directory of json files
            s_rank (int): Schmidt rank between 1 and 4, default: 0, 
            number_of_data_points (int): Number of training data points between 1 and 4, default: 0
            datatype (int): Data type between 1 and 4, default: 0
            metric_name (String): name of metric, default: TASC
            median (Boolean): if true, median metric values are used, default: False
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
        landscapes = []
        for run in range(len(data)):
            metric_values = data[run][label][landscape_label]
            landscapes.append(metric_values)
        if median:
            boxplot_dict[value] = np.median(landscapes, axis=0).flatten().tolist()
        else:
            boxplot_dict[value].extend(np.asarray(landscapes).flatten().tolist())

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


    # make boxplot 
    file_name = f"{dimension}D_{label}_boxplot_fixed_{file_name_string}.pdf"
    if median:
        plot_directory = f"/final_experiment/boxplots/{dimension}D_cost/variable_attributes/median_{label}_values/{variable_attribute}"
        title = f"{dimension}D cost function: median {label} values for\n{title_string}"
    else:
        title = f"{dimension}D cost function: {label} values for\n{title_string}"
        plot_directory = f"/final_experiment/boxplots/{dimension}D_cost/variable_attributes/{label}_values/{variable_attribute}"

    make_boxplot(
        boxplot_dict,
        y_label=label,
        x_label=variable_attribute,
        path=plot_directory,
        filename=file_name,
        title=title
    )
    del all_data

def make_all_2fixed_boxplots(dimension, directory):
    '''
        Make all boxplots where two training data attributes are fixed.

        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): directory of json files
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
            get_boxplot_two_fixed_attributes(dimension, directory, s_rank=s, datatype=dt, median=True)
        for ndp in [1,2,3,4]:
            get_boxplot_two_fixed_attributes(dimension, directory, number_of_data_points=ndp, datatype=dt)
            get_boxplot_two_fixed_attributes(dimension, directory, number_of_data_points=ndp, datatype=dt, median=True)
    for s in s_rank_range:
        for ndp in [1,2,3,4]:
            get_boxplot_two_fixed_attributes(dimension, directory, s_rank=s, number_of_data_points=ndp)
            get_boxplot_two_fixed_attributes(dimension, directory, number_of_data_points=ndp, datatype=dt, median=True)

def get_boxplot_per_attribute_fixedDatatype(dimension, directory, attribute, value_label, no_outliers=False):
    '''
        Four Boxplots in one, one plot per data type.

        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): source directory for JSON files 
            attribute (String): attribute for boxplots, one of "number_of_data_points" or "s_rank". 
            value_label (String): metric that is used to analyze cost landscape, one of ["TASC", "TSC", "MASC", "MSC"], 
            no_outliers (Boolean): if true, boxplots are without outliers, default: False
    '''
    assert dimension in [3,6]
    assert attribute in ["number_of_data_points", "s_rank"]
    assert value_label in ["TASC", "TSC", "MASC", "MSC"]
    # get data from json files
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
        print_relative_changes(data_dict[datatype],attribute, decimal=3)

    # Make Plots
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
    plt.savefig(f"{sub_directory}/{file_name}.pdf", format='pdf')  
    del all_data


######### Thesis Experiment Results Plots ######### 
def get_boxplot_perCostLandscape_per_attribute(dimension, directory, attribute, value_label="TASC", value_label_dict="std", no_outliers=False):
    '''
        Create Boxplot about std/median/mean of 'value_label' values per value of an 'attribute'.
        Used for Thesis.

        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): directory of json files
            attribute (String): "datatype", "number_of_data_points" or "s_rank". Attribute for x-axis of boxplot
            value_label (String): "TASC", "TSC", "MASC", "MSC". Values for Boxplot
    '''
    assert dimension in [3,6]
    assert attribute in ["datatype", "number_of_data_points", "s_rank"]
    assert value_label in ["TASC", "TSC", "MASC", "MSC"]
    assert value_label_dict in ["std", "median", "mean"]

    # get data from json files
    value_label_dict_name = {"std":"STD", "median":"Median", "mean":"Mean"}
    config_lists = get_list_of_corresponding_configs(dimension,attribute)
    if dimension == 3:
        conf_id_list = range(0,32)
    else:
        conf_id_list = range(0,64)
    all_data = load_json_data(directory, conf_id_list=conf_id_list)
    data_dict = {}
    attribute_values = config_lists.keys()
    for value in attribute_values:
        data_dict[value] = []
        for config in config_lists[value]:
            for run in range(len(all_data[config])):
                std = np.asarray(all_data[config][run][value_label][value_label_dict])
                data_dict[value].append(std)

    # print Summary
    print(f"Summary of {value_label_dict} of {value_label} of Cost Landscape per values of {attribute}")
    print_relative_changes(data_dict,attribute=attribute)
    
    # make boxplot
    make_boxplot(
        data_dict,
        y_label=f"{value_label_dict_name[value_label_dict]} of {value_label}", 
        x_label=attribute_names[attribute],
        path=f"/final_experiment/boxplots/{dimension}D_cost/{value_label_dict}",
        filename=f"{dimension}D_{value_label}_{value_label_dict}_boxplot_per_{attribute}",
        no_outliers=no_outliers
        )
    del all_data

def get_boxplot_fourier_density_per_attribute(dimension, directory, attribute):
    '''
        Boxplot for Fourier Density of cost landscape. Used for confirmation of previous results and own experiment setup.
        Used for thesis.
        
        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): directory of json files
            attribute (String): "datatype", "number_of_data_points" or "s_rank". Attribute for x-axis of boxplot
            value_label (String): "TASC", "TSC", "MASC", "MSC". Values for Boxplot
    '''
    assert dimension in [3,6]
    assert attribute in ["number_of_data_points", "s_rank", "datatype"]

    # get data from json files
    config_lists = get_list_of_corresponding_configs(dimension,attribute)
    if dimension == 3:
        conf_id_list = range(0,32)
    else:
        conf_id_list = range(0,64)
    all_data = load_json_data(directory, conf_id_list=conf_id_list)
    data_dict = {}
    
    for value in config_lists.keys():
        data_dict[value] = []
        for config in config_lists[value]:
            for run in range(len(all_data[config])):
                fd = all_data[config][run]["fourier density"]
                data_dict[value].append(fd)
    
    # print summary
    print_relative_changes(data_dict,attribute,decimal=2)
    
    # make boxplot
    make_boxplot(
        data_dict,
        y_label="Fourier Density",
        x_label=attribute_names[attribute],
        path=f"/final_experiment/boxplots/{dimension}D_cost/fd",
        filename=f"{dimension}D_fourier_density_boxplot_per_{attribute}",
    ) 
    del all_data

def get_boxplot_per_attribute(dimension, directory, attribute, value_label, no_outliers=False):
    '''
        Creates one boxplot per value of attribute (of training data) over landscape metric values
        Used for thesis.

        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): directory of json files
            attribute (String): "datatype", "number_of_data_points" or "s_rank". Attribute for x-axis of boxplot
            value_label (String): metric, one of "TASC", "TSC", "MASC", "MSC". Values for Boxplot.
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
    print_relative_changes(data_dict, attribute)

    # make boxplot
    make_boxplot(
        data_dict,
        y_label=value_label,
        x_label=attribute_names[attribute],
        path=f"/final_experiment/boxplots/{dimension}D_cost/TASC_values/no_fixed_attributes",
        filename=f"{dimension}D_{value_label}_boxplot_per_{attribute}",
        no_outliers=no_outliers
    )

    del all_data
    
def analyze_fourier_coefficients(dimension, directory, attribute, upper_limit=0.1):
    '''
        Make one histogram of fourier coefficient distribution per configuration attribute (datatype, number of data points, Schmidt rank) 
        Histograms van have an upper_limit, outliers above upper_limit are plotted in bar furthest to the right.
        Used for thesis (3D cost landscapes).

        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): directory of json files
            attribute (String): "datatype", "number_of_data_points" or "s_rank". Attribute for x-axis of boxplot
            upper_limit (float): upper limit of Histograms, default: 0.1
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
    

    n = len(data_dict.keys())
    n1 = 2
    n2 = 2
    if n==2:
        n2 = 1
    fig, axs = plt.subplots(n1, n2, figsize=(10,12))
    title_list = list(data_dict.keys())
    attribute_values = list(data_dict.keys())
    attribute_name = attribute_names[attribute]
    for i in range(n1):
        for j in range(n2):
            if n2 >1:
                ax = axs[i,j]
            else:
                ax = axs[i]
            index = 2*i + j
            value = attribute_values[index]
            title = f"{attribute_name} = {value}"
            current_fcoeff = np.asarray(data_dict[value])
            #counts, bins,_ = ax.hist(current_fcoeff,bins=100,range=(0,upper))
            print(f"{attribute_name} = {value}")
            outlier_aware_hist(ax, no_bins=100, data=current_fcoeff,lower=0, upper=upper_limit)
            x_ticks = np.linspace(0,upper_limit,num=7)
            #ax.set_xticks(x_ticks)
            ax.set_title(title, fontsize=14)
            ax.set_ylim(bottom=0, top=220)

    file_name = f"{dimension}D_fcoeff_hist_per_{attribute}"
    plt.tight_layout()
    plt.savefig(f"plots/final_experiment/boxplots/{dimension}D_cost/fcoeff/{file_name}.pdf", format='pdf')  
    plt.close()


    # determine how many FC are larger and smaller than upper_limit
    for val in attribute_values:
        print(f"{attribute}={val}")
        current_fcoeff = np.asarray(data_dict[val])
        n_smaller_eq = (current_fcoeff <= upper_limit).sum()
        n_larger = (current_fcoeff > upper_limit).sum()
        print(f"FC <= {upper_limit}:    {n_smaller_eq}")
        print(f"FC > {upper_limit}: {n_larger}")

    del all_data

def analyze_outliers(dimension, directory, attribute, value_label="TASC", no_outliers=False, z_score_thresh=3):
    '''
        Create Barplot of Number of TASC Outliers of each cost landscape per value of attribute.
        Used for thesis.

        Args:
            dimension (int): 3 or 6. Dimension of cost function
            directory (String): directory of json files
            value_label (String): metric value label in json file, one of "TASC", "TSC", "MASC", "MSC", default: "TASC"
            no_outliers (Boolean)
    '''
    assert dimension in [3,6]
    if dimension==3:
        config_range = range(0,32)
    else:
        config_range = range(0,64)
    all_data = load_json_data(directory, config_range, only_first_run=True)
    conf_lists = get_list_of_corresponding_configs(dimension, attribute)
    data_dict = {}
    data_dict_neg = {}
    data_dict_pos = {}
    attribute_values = conf_lists.keys()
    value_label_dict = f"{value_label} landscape".lower()
    for value in attribute_values:
        data_dict[value] = []
        data_dict_neg[value] = []
        data_dict_pos[value] = []
        for config in conf_lists[value]:
            for run in range(len(all_data[config])):
                #outliers = all_data[config][run][value_label]["outlier points"]
                landscape = np.asarray(all_data[config][run][value_label][value_label_dict])
                points = np.asarray(all_data[config][0]["points"])
                outliers,_,_,number_neg,number_pos = determine_outliers_in_grid(points, landscape, z_score_threshold=z_score_thresh)
                number_outliers = len(outliers)
                data_dict_pos[value].append(number_pos)
                data_dict_neg[value].append(number_neg)
                data_dict[value].append(number_outliers)
    # print summary and relative changes of boxplot (concrete values)
    print("Attribute:", attribute, "Z Score Threshold:", z_score_thresh)
    means, medians, stds = [], [], []
    means_neg, means_pos = [], []
    for value in data_dict.keys():
        means.append(np.round(np.mean(data_dict[value]),3))
        means_neg.append(np.round(np.mean(data_dict_neg[value]),3))
        means_pos.append(np.round(np.mean(data_dict_pos[value]),3))
        medians.append(np.round(np.median(data_dict[value]),3))
        stds.append(np.round(np.std(data_dict[value]),3))

    values = list(data_dict.keys())
    print(f"{attribute} = {values[0]}:     Mean = {means[0]}, Median = {medians[0]}, STD = {stds[0]}") 
    print(f"{attribute} = {values[0]}:     Mean neg = {means_neg[0]}, Mean pos = {means_pos[0]}")
    for i in range(1,len(means)):
        mean_diff = np.round((means[i-1]-means[i])/means[i-1]*100,1)
        mean_neg_diff = np.round((means_neg[i-1]-means_neg[i])/means_neg[i-1]*100,1)
        mean_pos_diff = np.round((means_pos[i-1]-means_pos[i])/means_pos[i-1]*100,1)
        median_diff = np.round((medians[i-1]-medians[i])/medians[i-1]*100,1)
        std_diff = np.round((stds[i-1]-stds[i])/stds[i-1]*100,1)
        print(f"{attribute} = {values[i]}:     Mean = {means[i]} -{mean_diff}%, Median = {medians[i]} -{median_diff}%, STD = {stds[i]} -{std_diff}%") 
        print(f"{attribute} = {values[i]}:     Mean neg = {means_neg[i]} -{mean_neg_diff}%, Mean pos = {means_pos[i]} -{mean_pos_diff}%")

    # plot: stack of average number of positive and negative outlier
    fig, ax = plt.subplots()
    x = [1,2,3,4]
    plt.bar(x, means_neg, color=low_color2, label='negative')
    plt.bar(x, means_pos, color=high_color2, label='positive', bottom=means_neg)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(data_dict.keys(), fontsize=14)
    attribute_name = attribute_names[attribute]
    ax.set_xlabel(attribute_name, fontsize=14)
    ax.set_ylabel("Average Number of Outliers", fontsize=14)
    file_name=f"{dimension}D_avgNumberOutliers_Bar_per_{attribute}_z={z_score_thresh}"
    plt.legend()
    plt.savefig(f"plots/final_experiment/boxplots/{dimension}D_cost/outlier/{file_name}.pdf", format='pdf')
    plt.close()

    del all_data


if __name__=="__main__":
    # example on how to use functions
    directory = "results/main_experiment/6D_cost"
    dimension = 6
    for attribute in ["s_rank", "number_of_data_points"]:
        get_boxplot_fourier_density_per_attribute(dimension, directory, attribute)
        
        

    