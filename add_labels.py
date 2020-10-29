import pandas as pd
import time
import sys


def main(data_filename, labels_filename, output):
    """
    load the datafile and the labels dictionary into a pandas dataframe
    """
    print("loading_data")
    data, dictionary = load_data(data_filename, labels_filename)
    print("cleaning data")
    icd_codes_only = clean_data(data)
    print("creating dictionary")
    dictionary = _create_dict(dictionary)
    print("getting categories")
    categories = create_categories(icd_codes_only, dictionary)
    #clear memory
    icd_codes_only = None
    dictionary = None
    print("cleaning categories")
    categories = clean_categories(categories)
    print("merge data")
    #append labels to end of full dataset
    data['label'] = categories
    print("write to csv")
    data.to_csv(output, index=False)

def load_data(data_filename, labels_filename):
    """
    Load the data and labels from csv
    """

    data = pd.read_csv(data_filename, dtype=str)
    labels = pd.read_csv(labels_filename, dtype=str)
    return data, labels

def clean_data(data):
    """
    return a dataframe with the ICD codes only to save memory
    """

    icd_codes_only = data.loc[:, 'Dx10_prin']

    return icd_codes_only

def _create_dict(dictionary):
    """
    convert the pandas dframe into a dictionary of codes and categories
    """
    dictionary = dictionary[['Code_10', 'Class_9_v6']]

    #trim the dict ICD10 codes to 4 to reduce unlabelled, spits a warning but works
    dictionary['Dx10_prin'] = dictionary['Code_10'].apply(lambda x: x[:4])
    dictionary = dictionary[['Dx10_prin', 'Class_9_v6']]

    #group by ICD10 (with 4 chars) and take the most common as the label
    dictionary = dictionary.groupby('Dx10_prin').Class_9_v6.agg(lambda x: x.mode()[0])
    dictionary = pd.DataFrame({'Dx10_prin':dictionary.index, 'Class_9_v6':dictionary.values})

    return dictionary

def create_categories(codes, dictionary):
    """
    return dframe of categories for each index of data
    """
    #reduce data code size to 4 chars to fit with dictionary
    codes = codes.str[:4]

    codes = codes.to_frame()
    #merge data and dictionary
    merged = codes.reset_index().merge(dictionary, how="left").set_index('index')

    #keep one label per index
    merged = merged.reset_index().drop_duplicates(subset='index').set_index('index').sort_index()

    return merged

def clean_categories(categories):
    """
    return category labels for dataset as a series
    convert Injury and Posion to one label
    change nan types to Other
    """
    list = ["Injury", "Poisoning"]
    list2 = ["nan"]
    categories[categories.isin(list)] = "Injury & Poisoning"
    categories[categories.isin(list2)] = "Other"

    categories = categories["Class_9_v6"]

    return categories


start_time = time.time()
#replace this with filepaths you want
main('/Users/adampapini/Downloads/Project/Data/B220_SAA_v1.csv','/Users/adampapini/Downloads/Project/Data/BIODS220_ICD_Dx_10_9_v7 - icd_dx_10_9_v7.csv', '/Users/adampapini/Downloads/Project/Data/labelled2.csv')
print("--- %s seconds ---" % (time.time() - start_time))

