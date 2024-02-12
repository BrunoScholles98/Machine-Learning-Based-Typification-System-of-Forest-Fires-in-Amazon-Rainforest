import geopandas as gpd
import numpy as np
import random
import pandas as pd
import pickle
import sys
import random

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


#######################################################################
# ---------------------------- OPEN DATA ---------------------------- #
#######################################################################
def open_data(train_path, test_path, last_index_train, last_index_test, features, target_name):
    
    if train_path[-3:] == 'csv':
        print('Inicializada a leitura dos Dataframes. O formato detectado é CSV.')
        train_df = pd.read_csv(train_path, sep = ';').iloc[:last_index_train]
        test_df = pd.read_csv(test_path, sep = ';').iloc[:last_index_test]

        train_df = train_df[features + target_name]
        test_df = test_df[features + target_name]
    
    elif train_path[-3:] == 'shp' and train_path == test_path:
        print('Inicializada a leitura dos Dataframes. O formato detectado é SHP. Possuímos dois SHPs iguais...')
        csvTotal = gpd.GeoDataFrame.from_file(train_path, geometry='geometry').iloc[:last_index_train]


        #csvTotal = pd.DataFrame(csvTotal)
        csvTotal[target_name] = csvTotal[target_name] - 1
        try:
            csvTotal.fire_type = csvTotal.fire_type.astype(int)
        except:
            csvTotal.fire_type_ = csvTotal.fire_type_.astype(int)        
        
        #if 'mv_indicadores_rotulado' not in train_path:
        csvTotal = csvTotal.sample(frac=1).reset_index(drop=True)
        partition_index = int(len(csvTotal) * 0.80)
        #else:
        #    partition_index = 289937


        train_df = csvTotal.iloc[:partition_index][features + target_name]
        test_df = csvTotal.iloc[partition_index:].reset_index()[features + target_name]
        

    elif train_path[-4:] == 'gpkg' and train_path == test_path:
        print('Inicializada a leitura dos Dataframes. O formato detectado é GPKGs. Possuímos dois GPKGs iguais...')
        csvTotal = gpd.GeoDataFrame.from_file(train_path, geometry='geometry')
        csvTotal = pd.DataFrame(csvTotal)
        csvTotal[target_name] = csvTotal[target_name] - 1
        try:
            csvTotal.fire_type = csvTotal.fire_type.astype(int)
        except:
            csvTotal.fire_type_ = csvTotal.fire_type_.astype(int)

        csvTotal = csvTotal.sample(frac=1).reset_index(drop=True)
        partition_index = int(len(csvTotal) * 0.8)
        train_df = csvTotal.iloc[:partition_index][features + target_name]
        test_df = csvTotal.iloc[partition_index:].reset_index()[features + target_name]    
        
    elif train_path[-3:] == 'shp' and train_path != test_path:
        print('Inicializada a leitura dos Dataframes. O formato detectado é SHP. Possuímos dois SHPs diferentes...')
        geo_df_train = gpd.GeoDataFrame.from_file(train_path, geometry='geometry')
        geo_df_test = gpd.GeoDataFrame.from_file(test_path, geometry='geometry')
        
        train_df = pd.DataFrame(geo_df_train).iloc[:last_index_train]
        test_df = pd.DataFrame(geo_df_test).iloc[:last_index_test]

        train_df = train_df[features + target_name]
        test_df = test_df[features + target_name]

    if 'id_event_1' in train_df.keys() and 'dt_maxima' in train_df.keys():   #util para LSTM
        train_df = train_df.sort_values(['id_event_1','dt_maxima'])
        test_df = test_df.sort_values(['id_event_1','dt_maxima'])
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return train_df, test_df

#######################################################################
# ---------------------------- NORMALIZE ---------------------------- #
#######################################################################
pathNorm = '/d01/ianporto/saves_lstm/norm_lstm_transpose.sav'
def normalizer_training(featuresStrings, norm, CSVtreino, CSVpred):
    """
    features: array com features do CSV fornecidos a serem normalizados
    featuresStrings: array com features do CSV fornecido que não devem ser normalizados
    norm: tecnica de normalizacao desejada
    CSVtreino, CSVpred: CSV fonte para a normalizacao

    Essa funcao recebe uma lista de features para normalizade do CSV e e uma lista de featureStrings para não normalizar
    as matrizes são combinadas utilizando pandas

    return dataframe, dataframe
    """
    #tecnicas = ['norm', 'standard', 'minmax', 'quantile', 'robust']


    strings = CSVtreino.columns.values.tolist()
    dftreino = CSVtreino[strings]
    dfpred = CSVpred[strings]

    for word in featuresStrings:
        strings.remove(word)

    features = strings

    X = dftreino[features]
    Xs =dftreino[featuresStrings]

    point = len(X)

    Y = dfpred[features]
    Ys = dfpred[featuresStrings]


    """dfX = pd.DataFrame(X)
    dfY = pd.DataFrame(Y)"""
        
    frames = [X, Y]
    Z = pd.concat(frames, ignore_index=True)
    #Z = Z.dropna() - não faz sentido remover linhas neste ponto, pois trara problemas na hora de remontar os csvs - se tem NaN aqui, o problema é mais em cima
    Z = Z.to_numpy()

    if norm == 'minmax':
        Z = MinMaxScaler().fit_transform(Z)
    elif norm == 'standard':
        scaler = StandardScaler()
        Z = scaler.fit_transform(Z)
        #pickle.dump(scaler, open(pathNorm, 'wb'))
    elif norm == 'norm':
        normalizer = Normalizer().fit(Z)  # fit does nothing
        #pickle.dump(normalizer, open(pathNorm, 'wb'))
        Z = normalizer.transform(Z)
    elif norm == 'quantile':
        scaler = QuantileTransformer(random_state=0)
        Z = scaler.fit_transform(Z)
    elif norm == 'robust':
        Z = RobustScaler().fit_transform(Z)
    else:
        print("ERRO - normalizacao desconhecida - tente: 'norm', 'standard', 'minmax', 'quantile', 'robust'")
        exit(1)
    
    Zdf = pd.DataFrame(Z)
    
    X = Zdf[:point]
    Y = Zdf[point:]
    Y = Y.reset_index(drop=True)
    X.columns = features
    Y.columns = features

    aux = pd.DataFrame(X)
    aux2 = pd.DataFrame(Xs)
    if(len(featuresStrings) > 0):
        treino = aux.join(aux2, lsuffix='_left', rsuffix='_right')
    else:
        treino = X

    aux = pd.DataFrame(Y)
    aux2 = pd.DataFrame(Ys)
    if(len(featuresStrings) > 0):
        pred = aux.join(aux2, lsuffix='_left', rsuffix='_right')
    else:
        pred = Y

    pred = pred.dropna()
    treino = treino.dropna()

    return [treino, pred]

#####################################################################
# ---------------------------- ONE HOT ---------------------------- #
#####################################################################
def last_index(input_list:list) -> int:
    return len(input_list) - 1

def oneHot_selected_columns(csv1,csv2,selected_collumns):
    onehoted_collumns = []

    frames = [csv1, csv2]

    csv = pd.concat(frames)
    csv = csv.reset_index()
    
    index_remover = csv.columns.values.tolist()
    index_remover.remove('index')
    csv = csv[index_remover]

    for selected_collumn in selected_collumns:
        #creating instance of one-hot-encoder
        encoder = OneHotEncoder(handle_unknown='ignore')
        #perform one-hot encoding on the selected column 
        encoder_csv = pd.DataFrame(encoder.fit_transform(csv[[selected_collumn]]).toarray())
        #merge one-hot encoded columns back with original DataFrame
        csv = csv.join(encoder_csv)

        strings = csv.columns.values.tolist()
        index = strings.index(0)

        i = 0
        while index <= last_index(strings):
            newName = selected_collumn + ' ' + str(i+1)
            csv = csv.rename(columns={i:newName})
            index += 1
            i += 1
            onehoted_collumns.append(newName)

    for i in selected_collumns:
        index_remover = csv.columns.values.tolist()
        index_remover.remove(i)
        csv = csv[index_remover]

    treino = csv.loc[(csv['Ano 7'] != 1)]
    predicao = csv.loc[(csv['Ano 7'] == 1)]

    predicao = predicao.reset_index()
    index_remover = predicao.columns.values.tolist()
    index_remover.remove('index')
    predicao = predicao[index_remover]

    return treino, predicao, onehoted_collumns
    

###########################################################################
# ---------------------------- BALANCEAMENTO ---------------------------- #
###########################################################################

def separate_samples(df, class_col, class_name, num_samples, random=False):
    # Create a new dataframe with only the rows where the class column matches the given class name
    class_df = df[df[class_col] == class_name]
    # Take the first 'num_samples' rows of the class dataframe
    if(random):
        return class_df.sample(n=num_samples) 
    samples_df = class_df.head(num_samples)
    return samples_df

def new_samples(df, class_col, class_name, num_samples, random=False):
    # Create a new dataframe with only the rows where the class column matches the given class name
    class_df = df.loc[df[class_col[0]] == class_name]
    # Take the first 'num_samples' rows of the class dataframe
    if(random):
        return class_df.sample(n=num_samples) 
    samples_df = class_df.head(num_samples)
    return samples_df

#percorre a serie puxando valores de mesmo id, puxando todos os items do mesmo evento
def series_traveler(geo_df, aux, n):

    if n < len(aux):
        aux = aux[0:n]
    else:
        n = len(aux)
    df = gpd.GeoDataFrame(columns = geo_df.columns)

    for i in range(n):
        aux_df = geo_df.loc[geo_df['id_event_1'] == aux[i]]
        df = pd.concat([df, aux_df])

    return df

def balance_geo_df(geo_df, class_col, n_samples_class0, n_samples_class1, n_samples_class2, n_samples_class3):
    type0_samples_df = separate_samples(geo_df, class_col, 0, n_samples_class0).reset_index(drop=True)
    type1_samples_df = separate_samples(geo_df, class_col, 1, n_samples_class1).reset_index(drop=True)
    type2_samples_df = separate_samples(geo_df, class_col, 2, n_samples_class2).reset_index(drop=True)
    type3_samples_df = separate_samples(geo_df, class_col, 3, n_samples_class3).reset_index(drop=True)

    balanced_df = pd.concat([type0_samples_df, type1_samples_df, type2_samples_df, type3_samples_df]).reset_index(drop=True)
    return balanced_df

#faz o sample de n_samples de cada classe do dataframe
def balance_time_series(geo_df, class_col, n_samples_class1, n_samples_class2, n_samples_class3, n_samples_class4):
    """     col = geo_df['id_event_1']
    aux = geo_df.dissolve(by='id_event_1')
    aux['id_event_1'] = col """

    aux1 = geo_df.loc[geo_df[class_col[0]] == 0]
    aux2 = geo_df.loc[geo_df[class_col[0]] == 1]
    aux3 = geo_df.loc[geo_df[class_col[0]] == 2]
    aux4 = geo_df.loc[geo_df[class_col[0]] == 3]

    aux1 = aux1['id_event_1'].unique()
    aux2 = aux2['id_event_1'].unique()
    aux3 = aux3['id_event_1'].unique()
    aux4 = aux4['id_event_1'].unique()
    #total = aux1[0]+aux2[0]+aux3[0]+aux4[0]
    #soma = n_samples_class1 + n_samples_class2 + n_samples_class3 + n_samples_class4


    """ type_0 = new_samples(aux, class_col, 1.0, n_samples_class0, random=True).reset_index(drop=True)
    type_1 = new_samples(aux, class_col, 2.0, n_samples_class1, random=True).reset_index(drop=True)
    type_2 = new_samples(aux, class_col, 3.0, n_samples_class2, random=True).reset_index(drop=True)
    type_3 = new_samples(aux, class_col, 4.0, n_samples_class3, random=True).reset_index(drop=True) """


    n_3 = len(aux3) #regra de tres...... n_3 eh o menor type e to com preguica de pensar em uma maneira de calcular o menor e fazer em ordem
    n_1 = int(n_3*n_samples_class1//(n_samples_class3))
    n_2 = int(n_3*n_samples_class2//(n_samples_class3))
    n_4 = int(n_3*n_samples_class4//(n_samples_class3))

    """ 
    if soma != 100 and soma != 1.0:
        print("Erro - a soma dos valores de balanceamento não formam 100% ou 1.0")
        sys.exit()

    if soma == 100:
        n_samples_class1 = n_samples_class1/100
        n_samples_class2 = n_samples_class2/100
        n_samples_class3 = n_samples_class3/100
        n_samples_class4 = n_samples_class4/100


    n_1 = np.floor(n_samples_class1*total)
    n_2 = np.floor(n_samples_class2*total)
    n_3 = np.floor(n_samples_class3*total)
    n_4 = np.floor(n_samples_class4*total)
    """

    samples_df_1 = series_traveler(geo_df, aux1, n_1)
    #print("fire_type = 1.0 pronto!!")
    samples_df_2 = series_traveler(geo_df, aux2, n_2)
    #print("fire_type = 2.0 pronto!!")
    samples_df_3 = series_traveler(geo_df, aux3, n_3)
    #print("fire_type = 3.0 pronto!!")
    samples_df_4 = series_traveler(geo_df, aux4, n_4)
    #print("fire_type = 4.0 pronto!!")

    """     type1_samples_df = series_traveler(type_1, geo_df)
    print("fire_type = 2.0 pronto!!")
    type2_samples_df = series_traveler(type_2, geo_df)
    print("fire_type = 3.0 pronto!!")
    type3_samples_df = series_traveler(type_3, geo_df)
    print("fire_type = 4.0 pronto!!") """

    samples_df = pd.concat([samples_df_1,samples_df_2,samples_df_3,samples_df_4])
    samples_df = samples_df.reset_index(drop=True)
    print("Dataset balanceado!!")

    return samples_df


def pack_dataset(dataset):
    ids = dataset['id_event_1'].tolist()
    id_list = list(set(ids))
    #id_list = list(ids.unique()) sugestão, sepa mais rapido
    random.shuffle(id_list)
    X = []
    for id in id_list:                                    # Une todas as detecções de um mesmo evento em sequência
        sub_df = dataset.loc[dataset['id_event_1'] == id]
        X.append(sub_df.values)

    return pd.DataFrame(X)

def separate_dataset(dataset, proportion=0.8):
    partition_index = int(len(dataset) * proportion)
    train_df = dataset.iloc[:partition_index].reset_index(drop=True)
    test_df = dataset.iloc[partition_index:].reset_index(drop=True)
    return train_df, test_df


