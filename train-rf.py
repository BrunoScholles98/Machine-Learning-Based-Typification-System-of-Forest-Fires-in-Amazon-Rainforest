import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

sys.path.append(str(Path(__file__).absolute().parents[1]))
import tools_rf

# Limpa o console
os.system('cls' if os.name == 'nt' else 'clear')

# Define o tamanho do dataset
LAST_INDEX_TRAIN = -1  # Tudo = -1
LAST_INDEX_TEST = -1  # Tudo = -1

# Define o path dos CSVs de treino e de predição
PATH_TREINO = '/d01/scholles/Codigos/maps/training_datasets/tb_eventos_train/tb_eventos_train.shp'
PATH_PREDICAO = '/d01/scholles/Codigos/maps/training_datasets/tb_eventos_train/tb_eventos_train.shp'

# Define onde o modelo treinado será salvo
PATH_TRAINED_MODEL = '/d01/scholles/Codigos/producao_files/model/RF_fire_type_PAPER.sav'

# Define as features e o target
FEATURES = ['persistenc','qtd_detecc','area_km2','areakm2_UC',
            'perc_in_UC','areakm2_IN','perc_in_IN','a_terr_pri','p_terr_pri','a_terr_pub',
            'p_terr_pub','a_terr_unk','p_terr_unk','area_CAR','per_CAR','deter_area','deter_days',
            'deter_perc','biomass_GS','biomass_AG','tree_cover','a_cover_0','p_cover_0','a_cover_1',
            'p_cover_1','a_cover_2','p_cover_2','a_cover_3','p_cover_3','a_cover_4','p_cover_4',
            'a_cover_5','p_cover_5','a_cover_6','p_cover_6','a_cover_7','p_cover_7','a_cover_8',
            'p_cover_8','a_cover_9','p_cover_9','a_cover_10','p_cover_10','a_cover_11',
            'p_cover_11','a_cover_12','p_cover_12','a_cover_13','p_cover_13','a_cover_14',
            'p_cover_14','a_cover_15','p_cover_15','a_cover_16','p_cover_16','a_cover_17',
            'p_cover_17','a_cover_18','p_cover_18','a_cover_19','p_cover_19','a_cover_20',
            'p_cover_20','a_cover_21','p_cover_21','a_cover_22','p_cover_22','a_cover_23',
            'p_cover_23','a_cover_24','p_cover_24','a_cover_25','p_cover_25','a_cover_26',
            'p_cover_26','a_cover_27','p_cover_27','a_cover_28','p_cover_28','a_cover_29',
            'p_cover_29','a_cover_30','p_cover_30','a_cover_31','p_cover_31','a_cover_32',
            'p_cover_32','a_cover_33','p_cover_33','a_cover_34','p_cover_34','a_cover_35',
            'p_cover_35','a_cover_36','p_cover_36','a_cover_37','p_cover_37','a_cover_38',
            'p_cover_38','a_cover_39','p_cover_39','a_cover_40','p_cover_40','a_cover_41',
            'p_cover_41','a_cover_42','p_cover_42','a_cover_43','p_cover_43','a_cover_44',
            'p_cover_44','a_cover_45','p_cover_45','a_cover_46','p_cover_46','a_cover_47',
            'p_cover_47','a_cover_48','p_cover_48','a_cover_49','p_cover_49','a_cover_50',
            'p_cover_50','a_cover_51','p_cover_51','a_cover_52','p_cover_52','a_cover_53',
            'p_cover_53','a_cover_54','p_cover_54','a_cover_55','p_cover_55','a_cover_56',
            'p_cover_56','a_cover_57','p_cover_57','a_cover_58','p_cover_58','a_cover_59',
            'p_cover_59','a_cover_60','p_cover_60','a_cover_61','p_cover_61','a_cover_62',
            'p_cover_62']

TARGET_NAME = ['fire_type']

# Abre os CSVs de treino e de predição
csvTreino, csvPredicao = tools_rf.open_data(PATH_TREINO, PATH_PREDICAO, LAST_INDEX_TRAIN, LAST_INDEX_TEST, FEATURES,
                                         TARGET_NAME)

X = csvTreino[FEATURES].to_numpy()
y = csvTreino[TARGET_NAME].to_numpy()

# Especifica o modelo de aprendizado de máquina e realiza o treinamento
learning_model = RandomForestClassifier(verbose=True)
print('Iniciado o treino!')
learning_model.fit(X, y.ravel())

# Salva o modelo treinado em disco
print('Iniciado o armazenamento do modelo treinado...')
pickle.dump(learning_model, open(PATH_TRAINED_MODEL, 'wb'))

# -------------------------------------------------------------------------------------------------------------#
# ------------------ A PARTIR DAQUI, TEMOS APENAS CÓDIGO DE VALIDAÇÃO PARA FINS ESTATÍSTICOS ------------------#
# -------------------------------------------------------------------------------------------------------------#

print('Iniciada predição!')
x_pred = csvPredicao[FEATURES].to_numpy()
y_pred = learning_model.predict(x_pred)
y_true = csvPredicao[TARGET_NAME].to_numpy()

print(classification_report(y_true, y_pred, digits=4))

from sklearn.metrics import confusion_matrix

# Crie a matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Calcule a acurácia para cada classe
accuracy_per_class = cm.diagonal() / cm.sum(axis=1)

# Printe a acurácia para cada classe
for i, accuracy in enumerate(accuracy_per_class):
    print(f"Acurácia para a classe {i+1}: {accuracy*100:.2f}%")