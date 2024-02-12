from pickle import NONE
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, Softmax, ReLU, Module, BatchNorm1d, modules, functional as F
from torch.nn.functional import one_hot
from torch.optim import Adam
from torchmetrics import Precision, Recall, F1Score, Accuracy
import torchsummary
from contextlib import redirect_stdout
import tqdm
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parents[1]))
import tools_mlp
from pathlib import Path

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of neurons in the Layers
LAYER1 = 512
LAYER2 = 256
LAYER3 = 128
LAYER4 = 64

# Training
LOG_INTERVAL = 1000
EPOCHS = 100
BATCH_SIZE = 256

PATH_TREINO = '/d01/scholles/Codigos/maps/training_datasets/tb_eventos_train/tb_eventos_train.shp'
PATH_PREDICAO = '/d01/scholles/Codigos/maps/training_datasets/tb_eventos_train/tb_eventos_train.shp'
LOAD_CHECKPOINT_FILENAME = ''
OUTPUT_PATH = Path('/d01/scholles/Codigos/censipam_type_logs/TCC2/RELU_BATCH' + str(BATCH_SIZE) + '_net_' + str(LAYER1) + '_' + str(LAYER2) + '_' + str(LAYER3) + '_' + str(LAYER4))

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_FILENAME = OUTPUT_PATH / 'results.csv'
STATS_FILENAME = OUTPUT_PATH / 'stats.json'
TENSORBOARD_LOG = OUTPUT_PATH / 'log'
SAVE_CHECKPOINT_FILENAME = OUTPUT_PATH / 'training.ckpt'
SUMMARY_MODEL_FILENAME = OUTPUT_PATH / 'model_summary.txt'

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

# Dataset Size
LAST_INDEX_TRAIN = -1 #Tudo = -1
LAST_INDEX_TEST = -1 #Tudo = -1
VAL_SUBSET_SIZE = 100000 # Máx = 1133894
N_SAMPLES = -1 # -1 para utilizar FRAC no sample()
FRAC_SAMPLES = 1

DF_TRAIN, DF_TEST = tools_mlp.open_data(PATH_TREINO, PATH_PREDICAO, LAST_INDEX_TRAIN, LAST_INDEX_TEST, FEATURES, TARGET_NAME)

#Learning Rate
LR = 1e-4
BETAS = (0.9, 0.999)

# Focal Loss
GAMMA = 0 # gamma = 0 e vetor de pesos = None para BCE
WEIGHTS = None # None para BCE, int qualquer para inverso frequencia, array para pesos manual

# Classification constants
NUM_CLASSES = DF_TEST[TARGET_NAME].nunique()[0]
print(NUM_CLASSES, 'labels diferentes detectadas.')

# Checkpoint dict keys
EPOCH_KEY = 'last_epoch'
MODEL_KEY = 'model'
OPTIMIZER_KEY = 'optimizer'
TRAIN_ITER_KEY = 'last_train_iteration'
VAL_ITER_KEY = 'last_val_iteration'

class FocalLoss(modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class CSVDataset(Dataset):
    def __init__(self, file):
        df = file
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.FL1 = Linear(n_inputs, LAYER1)
        self.act1 = ReLU()
        self.BN1 = BatchNorm1d(LAYER1)
        self.FL2 = Linear(LAYER1, LAYER2)
        self.act2 = ReLU()
        self.BN2 = BatchNorm1d(LAYER2)
        self.FL3 = Linear(LAYER2, LAYER3)
        self.act3 = ReLU()
        self.BN3 = BatchNorm1d(LAYER3)
        self.FL4 = Linear(LAYER3, LAYER4)
        self.act4 = ReLU()
        self.BN4 = BatchNorm1d(LAYER4)
        self.FL5 = Linear(LAYER4, NUM_CLASSES)
        self.BN5 = BatchNorm1d(NUM_CLASSES)
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        inter_values = []
        x = self.FL1(x)
        inter_values.append(x)
        x = self.act1(x)
        inter_values.append(x)
        x = self.BN1(x)
        inter_values.append(x)
        x = self.FL2(x)
        inter_values.append(x)
        x = self.act2(x)
        inter_values.append(x)
        x = self.BN2(x)
        inter_values.append(x)
        x = self.FL3(x)
        inter_values.append(x)
        x = self.act3(x)
        inter_values.append(x)
        x = self.BN3(x)
        inter_values.append(x)
        x = self.FL4(x)
        inter_values.append(x)
        x = self.act4(x)
        inter_values.append(x)
        x = self.BN4(x)
        inter_values.append(x)        
        x = self.FL5(x)
        inter_values.append(x)
        x = self.BN5(x)
        inter_values.append(x)
        x = self.softmax(x)
        return x, inter_values    


def min_max_norm(x):
    x_max = x.max()
    x_min = x.min()
    x = (x - x_min)/(x_max-x_min)
    return x


def get_predictions_from_subset_by_class(model, dataloader, subset_size=0):
    # TODO: implementar um sample e controlar o seed como uma constante no início do código
    indexes = np.random.permutation(len(dataloader.dataset))
    if subset_size > 0:
        indexes = indexes[:subset_size]

    x_subset, y_subset = dataloader.dataset[indexes]
    x_subset = torch.Tensor(x_subset).to(DEVICE)
    y_subset = torch.Tensor(y_subset).to(DEVICE)

    index_by_class = []
    for idx in range(NUM_CLASSES):
        curr_indexes_group = torch.where(y_subset == idx)[0]
        index_by_class.append(curr_indexes_group)

    y_pred, _ = model(x_subset)
    y_pred = y_pred.argmax(axis=1)
    pred_by_class = []
    for curr_indexes in index_by_class:
        curr_pred_group = y_pred[curr_indexes]
        pred_by_class.append(curr_pred_group)
    
    return pred_by_class


def train_one_step(model, optimizer, loss_fnc, x, y):
    optimizer.zero_grad()
    y_hat, inter_values = model(x)
    one_hot_y = one_hot(y.long(), num_classes=NUM_CLASSES)
    one_hot_y = torch.squeeze(one_hot_y, dim=1)
    loss = loss_fnc(y_hat, one_hot_y.float())
    loss.backward()
    optimizer.step()

    return y_hat, loss, inter_values


def train_by_one_epoch(model, optimizer, dataloader, all_steps_counter_train, writer, focal_loss_func):
    accuracy_fnc = Accuracy().to(DEVICE)
    training_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    training_bar.set_description("Training Progress (Epoch)")
    mean_loss_train = 0
    train_epoch_accuracy = 0

    for step_train,(x,y) in training_bar:
        x = x.cuda(DEVICE)
        y = y.cuda(DEVICE)
        y_hat, loss, inter_values = train_one_step(model, optimizer, focal_loss_func, x, y)
        mean_loss_train += loss
        
        y_hat_bin = y_hat.argmax(dim=1)[..., None]
        training_iteration_accuracy = accuracy_fnc(y_hat_bin.long(), y.long())
        train_epoch_accuracy += training_iteration_accuracy

        if step_train % LOG_INTERVAL == 0:
            writer.add_histogram('Train/Class_Prediction', y_hat_bin, all_steps_counter_train)
            writer.add_histogram('Train/GroundTruth', y, all_steps_counter_train)
            writer.add_scalar('Train/Iteration_Loss', loss, all_steps_counter_train)
            writer.add_scalar('Train/Iteration_Accuracy', training_iteration_accuracy, all_steps_counter_train)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram('Gradients/' + name, param.grad, all_steps_counter_train)
                writer.add_histogram('Weights/' + name, param.data, all_steps_counter_train)
            for i, out in enumerate(inter_values):
                writer.add_histogram('Inter_Outputs/' + str(i), out[i], all_steps_counter_train)
        
        all_steps_counter_train += 1
    
    mean_loss_train /= len(dataloader)
    train_epoch_accuracy /= len(dataloader)
    
    return all_steps_counter_train, mean_loss_train, train_epoch_accuracy


def validate_model(model, dataloader, all_steps_counter_val, writer, focal_loss_func):
    accuracy_fnc = Accuracy().to(DEVICE)
    validation_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    validation_bar.set_description("Validation Progress (Epoch)")
    mean_loss_validation = 0
    val_epoch_accuracy = 0

    with torch.no_grad():
        for validation_step, (x, y) in validation_bar:
            x = x.cuda(DEVICE)
            y = y.cuda(DEVICE)
            y_hat, _ = model(x)
            one_hot_y = one_hot(y.long(), num_classes=NUM_CLASSES)
            one_hot_y = torch.squeeze(one_hot_y, dim=1)
            loss_val = focal_loss_func(y_hat, one_hot_y.float())
            mean_loss_validation += loss_val

            y_hat_bin = y_hat.argmax(dim=1)[..., None]
            val_iteration_accuracy = accuracy_fnc(y_hat_bin.long(), y.long())
            val_epoch_accuracy += val_iteration_accuracy

            if validation_step % LOG_INTERVAL == 0:
                writer.add_histogram('Validation/Class_Prediction', y_hat_bin, all_steps_counter_val)
                writer.add_histogram('Validation/GroundTruth', y, all_steps_counter_val)
                writer.add_scalar('Validation/Iteration_Loss', loss_val, all_steps_counter_val)
                writer.add_scalar('Validation/Iteration_Accuracy', val_iteration_accuracy, all_steps_counter_val)
                
            all_steps_counter_val += 1
        
        mean_loss_validation /= len(dataloader)
        val_epoch_accuracy /= len(dataloader)

    return all_steps_counter_val, mean_loss_validation, val_epoch_accuracy


def run_train_on_all_epochs(model, optimizer, initial_epoch, initial_train_iter, initial_val_iter, train_dl, val_dl, focal_loss_func):
    writer = SummaryWriter(TENSORBOARD_LOG)
    epoch_bar = tqdm.tqdm(range(initial_epoch, EPOCHS), initial=initial_epoch, total=EPOCHS)
    epoch_bar.set_description("Training Progress (Overall)")

    all_steps_counter_train = initial_train_iter
    all_steps_counter_val = initial_val_iter
    
    for epoch in epoch_bar:

        all_steps_counter_train, mean_loss_train, train_epoch_accuracy = \
            train_by_one_epoch(model, optimizer, train_dl, all_steps_counter_train, writer, focal_loss_func)
        all_steps_counter_val, mean_loss_validation, val_epoch_accuracy = \
            validate_model(model, val_dl, all_steps_counter_val, writer, focal_loss_func)

        pred_by_class = get_predictions_from_subset_by_class(model, val_dl, subset_size=VAL_SUBSET_SIZE)

        writer.add_scalar('Train/Epoch_Loss', mean_loss_train, epoch)
        writer.add_scalar('Train/Epoch_Accuracy', train_epoch_accuracy, epoch)        
        writer.add_scalar('Validation/Epoch_Loss', mean_loss_validation, epoch)
        writer.add_scalar('Validation/Epoch_Accuracy', val_epoch_accuracy, epoch)
        for idx, curr_pred_group in enumerate(pred_by_class):
            writer.add_histogram('Validation/Class_Prediction_' + str(idx), curr_pred_group, epoch)

        torch.save({
            EPOCH_KEY: epoch,
            MODEL_KEY: model.state_dict(),
            OPTIMIZER_KEY: optimizer.state_dict(),
            TRAIN_ITER_KEY: all_steps_counter_train,
            VAL_ITER_KEY: all_steps_counter_val,
        }, SAVE_CHECKPOINT_FILENAME)


def test_and_log_metrics(model, dataloader):
    # Função customizada para gerar resultados finais e métricas sobre os resultados
    x, y = dataloader.dataset[:]
    x = torch.Tensor(x).to(DEVICE)
    y = torch.Tensor(y).to(DEVICE).long()

    y_pred, _ = model(x)
    y_pred = y_pred.argmax(axis=1)[..., None]

    accuracy = Accuracy(num_classes=NUM_CLASSES, average=None).to(DEVICE)(y_pred, y)
    precision = Precision(num_classes=NUM_CLASSES, average=None).to(DEVICE)(y_pred, y)
    recall = Recall(num_classes=NUM_CLASSES, average=None).to(DEVICE)(y_pred, y)
    f1 = F1Score(num_classes=NUM_CLASSES, average=None).to(DEVICE)(y_pred, y)
    support = torch.unique(y, return_counts=True)[1]

    weighted_accuracy, weighted_precision, weighted_recall, weighted_f1 = 0, 0, 0, 0
    report = []
    for i in range(NUM_CLASSES):
        weight = support[i] / len(y)
        weighted_accuracy += weight * accuracy[i]
        weighted_precision += weight * precision[i]
        weighted_recall += weight * recall[i]
        weighted_f1 += weight * f1[i]
        report.append(['class_' + str(i), accuracy[i].item(), precision[i].item(), recall[i].item(), f1[i].item(), support[i].item()])
    report.append(['weighted', weighted_accuracy.item(), weighted_precision.item(), weighted_recall.item(), weighted_f1.item(), len(y)])

    report_df = pd.DataFrame(report, columns=['-', 'accuracy', 'precision', 'recall', 'f1', 'support'])
    report_df.set_index(report_df.columns[0])
    print('**************************************** REPORT ****************************************')
    print(report_df.to_markdown(index=False))
    report_df.to_csv(STATS_FILENAME, sep = ';', index=False)

    pred_df = pd.DataFrame(torch.concat([y, y_pred], axis=1).cpu(), columns=['GT', 'predictions'])
    pred_df.to_csv(RESULTS_FILENAME, sep = ';', index = False)


def init_or_load_checkpoint():
    global EPOCHS

    model = MLP(len(FEATURES))
    epochs_already_run = 0
    initial_train_iter = 0
    initial_val_iter = 0
    optimizer = Adam(model.parameters(), lr=LR, betas=BETAS)
    if len(LOAD_CHECKPOINT_FILENAME) > 0:
        if Path(LOAD_CHECKPOINT_FILENAME).exists():
            print("****************************************Checkpoint carregado!****************************************")
            checkpoint = torch.load(LOAD_CHECKPOINT_FILENAME)
            model.load_state_dict(checkpoint[MODEL_KEY])
            epochs_already_run = checkpoint[EPOCH_KEY] + 1 # + 1 porque o contador de épocas começa em 0
            # EPOCHS representará o número adicional de épocas que serão rodadas (mais fácil o uso assim)
            # Se EPOCHS = 0, então o modelo será carregado e só será testado
            EPOCHS += epochs_already_run
            initial_train_iter = checkpoint[TRAIN_ITER_KEY] + 1
            initial_val_iter = checkpoint[VAL_ITER_KEY] + 1
            optimizer.load_state_dict(checkpoint[OPTIMIZER_KEY])
            print("Este checkpoint representa um modelo já treinado {:d} épocas".format(epochs_already_run))

            # workaround to a problem of loading the state of an optimizer into GPU
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(DEVICE)
        else:
            print("****************************************Checkpoint nao encontrado!****************************************")
            raise FileNotFoundError("O ARQUIVO DE CHECKPOINT PARA CARREGAMENTO NÃO FOI ENCONTRADO!")
    
    model.to(DEVICE)

    with open(SUMMARY_MODEL_FILENAME, 'w') as f:
        with redirect_stdout(f):
            torchsummary.summary(model, (len(FEATURES),), BATCH_SIZE, DEVICE.type)
    print("****************************************Model Summary!****************************************")
    torchsummary.summary(model, (len(FEATURES),), BATCH_SIZE, DEVICE.type)
    
    return model, optimizer, epochs_already_run, initial_train_iter, initial_val_iter


def main():
    print('Inicializada a normalização geral dos CSVs...')
    csvTreino, csvPredicao = tools_mlp.normalizer_training(TARGET_NAME, 'standard', DF_TRAIN, DF_TEST)

    if  N_SAMPLES > 0:
        csvTreino = csvTreino.sample(n = N_SAMPLES)
        csvPredicao = csvPredicao.sample(n = N_SAMPLES)
    else: 
        csvTreino = csvTreino.sample(frac = FRAC_SAMPLES)
        csvPredicao = csvPredicao.sample(frac = FRAC_SAMPLES)

    global WEIGHTS
    if not isinstance(WEIGHTS, type(None)):
        if isinstance(WEIGHTS, int):
            count_classes = csvTreino[TARGET_NAME].value_counts().sort_index()
            classes_freq = count_classes / count_classes.sum()
            inv_freq = 1/classes_freq
            WEIGHTS = (inv_freq/inv_freq.sum()).to_numpy()
        WEIGHTS = torch.Tensor(WEIGHTS).to(DEVICE)
    focal_loss_func = FocalLoss(weight=WEIGHTS, gamma=GAMMA)

    # TODO: implementar ideia de treinamento, validação e teste com subsets diferentes
    train_dl = DataLoader(CSVDataset(csvTreino), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(CSVDataset(csvPredicao), batch_size=BATCH_SIZE, shuffle=False)
    model, optimizer, initial_epoch, initial_train_iter, initial_val_iter = init_or_load_checkpoint()
    
    print('****************************************Iniciado o treinamento!****************************************')
    run_train_on_all_epochs(model, optimizer, initial_epoch, initial_train_iter, initial_val_iter, train_dl, val_dl, focal_loss_func)
    print('****************************************Treinamento Encerrado!****************************************')

    print('****************************************Iniciado o teste!****************************************')
    test_and_log_metrics(model, val_dl)
    print('****************************************Encerrado o teste!****************************************')


if __name__ == '__main__':
    main()