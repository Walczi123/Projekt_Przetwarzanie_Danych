import locale
from traceback import format_exc
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
from math import floor
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
plt.style.use('ggplot')
RANDOM_SEED = 0

def read_transport_data(filename):
    types_mapping = {
        'Numer / oznaczenie linii' : str,                                     
        'Wariant linii' : str,                                              
        'Kurs' : str,                                                        
        'Oznaczenie brygady' : str,
        'Czas rzeczywisty odjazdu wg. rozkładu' : str,                              
        'Czas odjazdu z przystanku' : str,
    }

    df = pd.read_csv(filename, sep=';', dtype=types_mapping)
    df = df.drop(columns='Data')
    
    names_mapping = {
        'Numer / oznaczenie linii' : 'Nr linii',                                     
        # 'Wariant linii'                                                    
        # 'Kurs'                                                                
        # 'Oznaczenie brygady'                                                  
        # 'Numer taborowy'                                                     
        # 'Nazwa przystanku'                                                    
        # 'Numer przystanku'                                                   
        'Czas rzeczywisty odjazdu wg. rozkładu' : 'Rzeczywisty czas odjazdu',                              
        'Czas odjazdu z przystanku' : 'Rozkładowy czas odjazdu',                                      
        'Liczba pasażerów w pojeździe przed przystankiem (dane skorygowane)' : 'Liczba pasaz. przed',  
        'Liczba osób które wysiadły (dane skorygowane)' : 'Liczba wysiadających pasaz.',                       
        'Liczba osób które wsiadły (dane skorygowane)' : 'Liczba wsiadajacych pasaz.',                     
        'Liczba pasażerów w pojeździe po odjeździe (dane skorygowane)' : 'Liczba pasaz. po odjezdzie'
    }
    df.rename(columns=names_mapping, inplace=True)
    return df

def check_data(df:pd.DataFrame):
    counter = 0
    counter += len(df.loc[df['Liczba pasaz. przed'] - df['Liczba wysiadających pasaz.'] + df['Liczba wsiadajacych pasaz.'] != df['Liczba pasaz. po odjezdzie']])
    counter += len(df[df['Liczba pasaz. przed'] < 0])
    print(f'Amount of invalid data: {counter}')

def remove_errors(df:pd.DataFrame):
    df.drop(df[df['Liczba pasaz. przed'] < 0].index, inplace=True) # remove invlid amout of passengers
    df.drop(df[df['Kurs'].str[:2] > '23'].index, inplace= True)
    df.dropna(inplace=True)

def make_date_one_type(x):
    return pd.Timestamp(x)

def parse_data(df:pd.DataFrame):
    df['Kurs'] = df['Kurs'].apply(lambda x: make_date_one_type(x))
    df['Rzeczywisty czas odjazdu'] = df['Rzeczywisty czas odjazdu'].apply(lambda x: make_date_one_type(x))
    df['Rozkładowy czas odjazdu'] = df['Rozkładowy czas odjazdu'].apply(lambda x: make_date_one_type(x))

    df['Godzina odjazdu'] = df['Rozkładowy czas odjazdu'].apply(lambda x: x.hour + x.minute/60 + x.second/3600)
    df['Godzina odjazdu przedział'] = df['Rozkładowy czas odjazdu'].apply(lambda x: x.hour * 4 + floor(x.minute/15))
    df['Opóźnienie w minutach'] = (df['Rzeczywisty czas odjazdu'] - df['Rozkładowy czas odjazdu']).apply(lambda x: int(x.total_seconds()//60))

def add_type(df:pd.DataFrame):
    trans_type = []
    for nr in df['Nr linii']:
        if nr.isnumeric():
            trans_type.append('autobus')
        elif nr[0] == 'N':
            trans_type.append('nocny')
        elif nr[0] == 'Z':
            trans_type.append('zastepczy')
        else:
            trans_type.append('inny')
    df['Typ'] = trans_type

def get_data(path:str):
    df = read_transport_data(path)
    check_data(df)
    remove_errors(df)
    parse_data(df)
    add_type(df)
    return df

def drop_other_types(df:pd.DataFrame):
    df_no_type = df.drop(df[df['Typ'] != 'autobus'].index, inplace= False)
    df_no_type.drop(columns=['Typ'], inplace= True)
    return df_no_type

def create_heatmap(df, corr):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

def find_clusters(speeds, iterations, df, column, name = "Zatłoczenie"):
    k = len(speeds)
    data = np.array(df[column]).reshape(1,-1).T
    clusters = KMeans(
        n_clusters = k,
        max_iter = iterations,
        random_state = RANDOM_SEED
    ).fit(data)
    
    cc, *_ = clusters.cluster_centers_.reshape(1,k)
    cc_ord = np.argsort(cc)
    class_mapping = dict(zip(cc_ord, speeds))
    
    df['class'] = clusters.labels_
    df['class'] = df['class'].map(class_mapping)
    
    return df

def get_clustering_dict(df:pd.DataFrame):
    df_unique = np.unique(df[["Liczba pasaz. przed", 'class']], axis=0)
    classes = [0,1,2]
    d = dict()
    for c in classes:
        for x in [x[0] for x in df_unique if x[1] == c]:
            d[x] = c
    return d

# CROWDING = ['pusto','małe zatłoczenie', 'średnie zatłoczenie', 'duze zatłoczenie', 'olbrzymie zatłoczenie']
CROWDING = [0,1,2]

def get_clustered_data_with_bus_only(path:str, crowding:list = CROWDING, colum = "Liczba pasaz. przed" ):
    df = get_data(path)
    df = drop_other_types(df)
    df = find_clusters(crowding, 10, df, colum)
    clustering_dict = get_clustering_dict(df)
    return df, clustering_dict

def get_numeric_name(df:pd.DataFrame, column:str, drop:bool = False):
    stations_set = set(df[column].values)
    stations_iterator =  set(range(1,len(stations_set)+1))
    station_dict = dict(zip(stations_set, stations_iterator))

    df[f"Numeryczna {column}"] = df[column].map(station_dict).astype(int)

    if drop:
        df.drop(columns=[column], inplace= True)

    return df

def split_into_lines(df:pd.DataFrame, drop_line:bool = False, lower_bound:int = 4000):
    lines  = df['Nr linii'].unique()
    d = dict()
    for line in lines:
        tmp_df = df[df['Nr linii'] == line]
        if len(tmp_df) > lower_bound:
            if drop_line:
                tmp_df.drop(columns=['Nr linii'], inplace= True)
            d[line] = tmp_df
    return d

def get_train_and_test_by_lines(df:pd.DataFrame, X_columns:list, y_columns:list, lines:list = None, lower_bound:int = 4000,  test_size=0.1, random_state=1):
    d_lines = split_into_lines(df, lower_bound=lower_bound)
    if lines is not None:
        for k in d_lines.keys:
            if not k in lines:
                del d_lines[k]
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []
    for line in d_lines.values():
        X = line[X_columns]
        y = line[y_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    
    df_x_train = pd.concat(X_trains)
    df_x_test = pd.concat(X_tests)
    df_y_train = pd.concat(y_trains)
    df_y_test = pd.concat(y_tests)

    return df_x_train, df_x_test, df_y_train, df_y_test

def get_train_and_test_for_lines(line:pd.DataFrame, X_columns:list, y_columns:list, test_size=0.1, random_state=RANDOM_SEED):
    X = line[X_columns]
    y = line[y_columns]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def plot_classification_report(y_tru, y_prd, figsize=(10, 10), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score'] #, 'support']
    yticks = list(np.unique(y_tru))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep[:, :-1],
                annot=True,
                cbar=False,
                xticklabels=xticks,
                yticklabels=yticks,
                ax=ax)

def classification_of_list(list:list, clastering_dict:dict):
    return [clastering_dict[l] for l in list]

def regression_to_classification(y, prediceted, clastering_dict:dict):
    return [clastering_dict[l] for l in prediceted], [clastering_dict[l] for l in y]

def accuracy_of_regression(y, prediceted, clustering_dict:dict):
    prediceted, y = regression_to_classification(y, prediceted, clustering_dict)
    print("Accuracy: {:.5f}".format(accuracy_score(y, prediceted)))
    plot_classification_report(y, prediceted)