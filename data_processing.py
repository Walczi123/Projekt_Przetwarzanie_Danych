from sklearn import svm
import seaborn as sns
import locale

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
import seaborn as sns

from datetime import datetime


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

def make_date_one_type(x):
    return pd.Timestamp(x)

def show_me_outliers(data, nu, kernel, gamma):
    X = data
    w = len(nu)*len(gamma)*len(kernel)
    ins_outs = np.zeros((w, 2))
    k=0
    for i in range(len(nu)):
        for j in range(len(kernel)):
            for z in range(len(gamma)):
                clf2 = svm.OneClassSVM(nu = nu[i], gamma = gamma[z], kernel = kernel[j])
                #clf.fit(X, y)
                #1 for inliers, -1 for outliers
                data['nu='+str(nu[i])+', gamma='+str(gamma[z])+', ' + str(kernel[j])] = pd.DataFrame(clf2.fit_predict(X))
                hue_col = str('nu='+str(nu[i])+', gamma='+str(gamma[z])+', ' + str(kernel[j]))
                plt.figure(hue_col, figsize = (10,10))
                sns.scatterplot(data = data, x = 'Godzina odjazdu', y = 'Liczba pasaz. po odjezdzie' , hue = hue_col)
                
                stats = np.unique(clf2.fit_predict(X), return_counts=True)
                ins_outs[k,0] = stats[1][0]
                ins_outs[k,1] = stats[1][1]
                
                k+=1

    ins_outs = pd.DataFrame(ins_outs, columns = stats[0], index = data.columns.values[-w:]).astype(int)         
    return data, ins_outs

df = read_transport_data('./data/SZP-2021-09-03.csv')

df.drop(df[df['Liczba pasaz. przed'] < 0].index, inplace=True) # remove invlid amout of passengers

df.drop(df[df['Kurs'].str[:2] > '23'].index, inplace= True)

is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df[row_has_NaN]

df.drop(df[row_has_NaN].index, inplace=True)

df['Kurs'] = df['Kurs'].apply(lambda x: make_date_one_type(x))
df['Rzeczywisty czas odjazdu'] = df['Rzeczywisty czas odjazdu'].apply(lambda x: make_date_one_type(x))
df['Rozkładowy czas odjazdu'] = df['Rozkładowy czas odjazdu'].apply(lambda x: make_date_one_type(x))

df['Godzina odjazdu'] = df['Rozkładowy czas odjazdu'].apply(lambda x: x.hour + x.minute/60 + x.second/3600)

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

dat = df[["Liczba pasaz. po odjezdzie", "Godzina odjazdu"]]

nu = [0.1]
gamma = [0.1]
kernel = ['rbf']
#‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’, default = 'rbf'
df2 = dat[:1000]
df2, ins_out = show_me_outliers(data = df2, nu = nu, kernel = kernel, gamma = gamma)
print('end')