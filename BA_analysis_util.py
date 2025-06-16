import json
import os
from matplotlib import pyplot as plt
import numpy as np

from BA_experiment_resources import attribute_configs_3D, attribute_configs_6D, attribute_values_3D, attribute_values_6D, attribute_names

# colors from Viridis colormap
low_color = "#440154" # dark purple
mid_color = "#21918c" # teal
high_color = "#fde725" # yellow

low_color2 = "#5daea0" # PineGreen (60%) (LaTeX)
high_color2 = "#d17b5f" # BrickRed (60%) (LaTeX)

######### Helper Functions ######### 
def determine_index(grid_points, point):
    '''
        Determine index of point within a grid.
        Args:
            grid_points (list): list of points in grid in each dimension
            point (list): point in grid
        Returns
            tuple: index of point as tuple
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
        Args:
            dimension (int): 3 or 6, 
            s_rank (int): Schmidt rank between 1 and 4, default: 0, 
            number_of_data_points (int): Number of training data points between 1 and 4, default: 0
            datatype (int): Data type between 1 and 4, default: 0
        Returns:
            dict: Keys are values of variable attribute, values are corresponding config_ids (int)
    '''
    assert dimension in [3,6]
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

def check_if_correct_configs():
    '''
        Used to test get_list_of_corresponding_configs. 
    '''
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

def load_json_data(directory, conf_id_list=range(0,64), only_first_run=False):
    '''
        Load JSON-data for each config_id in conf_id_list from files saved in directory.
        File names start with "conf_{config_id}_" and end with ".json".
        Default conf_id_list is all configs, i.e. 0 to 319.

        Args:
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


######### Helper Functions for Plots ######### 
def print_relative_changes(data_dict, attribute, decimal=1):
    '''
        Print summary of data and relative changes between data sets (mean, median, std)
        Args:
            data_dict (dict): dictionary of values
            attribute (String): name of attribute (keys of data_dict are values of attribute)
            decimal (int): number if decimal points, default: 1
    '''
    print("Attribute:", attribute)
    means, medians, stds = [], [], []
    for value in data_dict.keys():
        means.append(np.round(np.mean(data_dict[value]),decimal))
        medians.append(np.round(np.median(data_dict[value]),decimal))
        stds.append(np.round(np.std(data_dict[value]),decimal))

    values = list(data_dict.keys())
    print(f"{attribute} = {values[0]}:     Mean = {means[0]}, Median = {medians[0]}, STD = {stds[0]}") 
    for i in range(1,len(means)):
        mean_diff = np.round((means[i-1]-means[i])/means[i-1]*100,decimal)
        median_diff = np.round((medians[i-1]-medians[i])/medians[i-1]*100,decimal)
        std_diff = np.round((stds[i-1]-stds[i])/stds[i-1]*100,decimal)
        print(f"{attribute} = {values[i]}:     Mean = {means[i]} -{mean_diff}%, Median = {medians[i]} -{median_diff}%, STD = {stds[i]} -{std_diff}%") 
         
def make_boxplot(data_dict, y_label, x_label, path, filename, title="",no_outliers=False):
    '''
        Make boxplots of values in data_dict (for thesis).
        Args:
            data_dict (dict): values for boxplots, one boxplot per key of data_dict
            y_label (String): label of x_axis
            x_label (String): label of x_axis
            path (String): path for plot (must start with /, end without /), if "" file is saved unter "plots/"
            title (String): title of plot, optional, default: ""
            no_outliers (Boolean): (default: False) if true: boxplot created without fliers, filename changed
    '''          

    fig, ax = plt.subplots()
    if no_outliers:
        ax.boxplot(data_dict.values(), showfliers=False)
        file_name = f"{filename}_noOutliers"
    else:
        ax.boxplot(data_dict.values())
        file_name = filename
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_xticklabels(data_dict.keys(), fontsize=14)
    #ax.set_title(f"{dimension}D cost function: {value_label} per {attribute}")
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    if len(title)>0:
        ax.set_title(title, fontsize=16)
    os.makedirs(f"plots{path}", exist_ok=True)
    plt.savefig(f"plots{path}/{file_name}.pdf", format='pdf',bbox_inches='tight')

def outlier_aware_hist(ax, no_bins,data, lower=None, upper=None):
    '''
        Creates an outlier aware histogram, where x-axis is limited between lower and upper, and upper and lower outliers are appended to right and left of Histogram.
        Adapted from Source: https://stackoverflow.com/questions/15837810/making-pyplot-hist-first-and-last-bins-include-outliers
        Used for thesis (3D cost landscape).

        Args:
            ax (axis object): axis object that contains histogram
            no_bins (int): number of bins
            data (list): data for histogram
            lower (float): lower limit of histogram
            upper (float): upper limit of histogram
    '''
    if not lower or lower < data.min():
        lower = data.min()
        lower_outliers = False
    else:
        lower_outliers = True

    if not upper or upper > data.max():
        upper = data.max()
        upper_outliers = False
    else:
        upper_outliers = True

    n, bins, patches = ax.hist(data, range=(lower, upper), bins=no_bins,color=low_color2)

    if lower_outliers:
        n_lower_outliers = (data < lower).sum()
        patches[0].set_height(patches[0].get_height() + n_lower_outliers)
        patches[0].set_facecolor(low_color2)
        patches[0].set_label('Lower outliers: ({:.3f}, {:.3f})'.format(data.min(), lower))

    if upper_outliers:
        n_upper_outliers = (data > upper).sum()
        print(n_upper_outliers)
        patches[-1].set_height(patches[-1].get_height() + n_upper_outliers)
        patches[-1].set_facecolor(high_color2)
        patches[-1].set_label('Upper outliers: ({:.3f}, {:.3f})'.format(upper, data.max()))

    if lower_outliers or upper_outliers:
        ax.legend()
    ax.set_xlabel("Absolute value of Fourier coefficient", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12) 
