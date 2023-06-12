from locale import normalize
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import seaborn as sn
import pandas as pd
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
sys.path.append('./scripts/')
from data import *
from model import SpeakerClassifier
from loss import *
from utils import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
#from torchmetrics import ConfusionMatrix

path_test_file = "/home/usuaris/veu/david.linde/features/ca_es_en/test.lst"
path_net = "/home/usuaris/veu/david.linde/DoubleAttentionSpeakerVerification/scripts/models/ca_es_en_load/resnet_resnet_1.0_128batchSize_0.0001lr_0.001weightDecay_512kernel_400embSize_30.0s_0.4m_DoubleMHA_32_30000.chkpt"
name = path_net.split("/")[-2]
classes = ('teens','twenties','thirties','fourties','fifties','sixties','seventies','eighties','nineties')
new_classes =  ('twenties','thirties','fourties','fifties','sixties','seventies')

class Evaluator:

    def __init__(self, params, device):

        self.params = params
        self.device = device
        self.__load_network()
        self.__load_data()



    def __load_data(self):
        with open(path_test_file, 'r') as data_test_labels_file:
            test_labels=data_test_labels_file.readlines()
                
        data_loader_parameters = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
        self.test_generator = DataLoader(Dataset(test_labels, self.params), **data_loader_parameters)

    def __load_network(self):

        self.net = SpeakerClassifier(self.params, self.device)
        print(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        checkpoint = torch.load(path_net)
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.params = checkpoint['settings']
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = nn.DataParallel(self.net)

    def evaluator(self):
        test_accuracy = 0
        test_batch = 0
        predictions_fscore = []
        labels_fscore = []
        with torch.no_grad():
            valid_time = time.time()
            self.net.eval()
            iterator = iter(self.test_generator)

            for input, labels in iterator:
                input, labels = input.float().to(self.device), labels.long().to(self.device)
                prediction, AMPrediction  = self.net(input, label=labels, step=self.step)
                predictions_fscore.append(prediction)
                labels_fscore.append(labels)
                test_accuracy += Accuracy(prediction, labels)
                test_batch +=1
                #print(test_batch)
            test_accuracy = test_accuracy * 100/test_batch
            predictions_fscore = torch.cat(predictions_fscore)
            labels_fscore = torch.cat(labels_fscore)
            pca(predictions_fscore,labels_fscore,name)
            #tsne(predictions_fscore,labels_fscore,name)
            centroid(predictions_fscore,labels_fscore,name)
            f1score = fscore(predictions_fscore,labels_fscore)
            f1score_modified = fscore_modified(predictions_fscore,labels_fscore)
            modified_predictions, modified_labels = modified_target(predictions_fscore,labels_fscore)
            mse = nn.MSELoss(reduction='none')
            error = mse(modified_predictions, modified_labels).sum(axis=1).mean()
            #rmse = mean_squared_error(modified_predictions,modified_labels)

            print('Fscore: {f1score: 3.3f}, FscoreMod: {f1score_modified: 3.3f}, MSE: {error: 3.3f}, Validation accuracy:{validation_accuracy: 3.3f}, elapse:{elapse: 3.3f} min'.format(f1score = f1score, f1score_modified=f1score_modified, error=error, validation_accuracy=test_accuracy, elapse=(time.time()-valid_time)/60))
            cf_matrix = confusion_matrix(labels_fscore.cpu().numpy(), torch.max(predictions_fscore, 1)[1].cpu().numpy(), normalize='true')
            df_cm = pd.DataFrame(cf_matrix, index = [i for i in new_classes],columns = [i for i in new_classes])
            plt.figure(figsize = (15,10))
            sn.heatmap(df_cm, annot=True)
            plt.title("Confusion Matrix - Test set")
            plt.xlabel("Predicted")
            plt.ylabel("True Label")
            plt.savefig(f"/home/usuaris/veu/david.linde/CommonVoice11/{name}.png")
def main(opt):

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    print('Defining Device')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    print('Loading Evaluator')
    evaluator = Evaluator(opt, device)
    evaluator.evaluator()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a VGG based Speaker Embedding Extractor')
   
    parser.add_argument('--train_data_dir', type=str, default='', help='data directory.')
    parser.add_argument('--valid_data_dir', type=str, default='', help='data directory.')
    parser.add_argument('--train_labels_path', type = str, default = 'labels/Vox2.ndx')
    parser.add_argument('--data_mode', type = str, default = 'normal', choices=['normal','window'])
    parser.add_argument('--validation_file', type = str, default='labels/validation.lst')
    parser.add_argument('--valid_clients', type = str, default='labels/clients.ndx')
    parser.add_argument('--valid_impostors', type = str, default='labels/impostors.ndx')
    parser.add_argument('--out_dir', type=str, default='./models/model1', help='directory where data is saved')
    parser.add_argument('--model_name', type=str, default='CNN', help='Model associated to the model builded')
    parser.add_argument('--front_end', type=str, default='resnet', choices = ['VGG3L','VGG4L','resnet'], help='Kind of Front-end Used')
    
    # Network Parameteres
    parser.add_argument('--window_size', type=float, default=3.5, help='number of seconds per window')
    parser.add_argument('--randomSlicing',action='store_true')
    parser.add_argument('--normalization', type=str, default='cmn', choices=['cmn', 'cmvn'])
    parser.add_argument('--kernel_size', type=int, default=512)
    parser.add_argument('--embedding_size', type=int, default=400)
    parser.add_argument('--heads_number', type=int, default=32)
    parser.add_argument('--pooling_method', type=str, default='DoubleMHA', choices=['Attention', 'MHA', 'DoubleMHA'], help='Type of pooling methods')
    parser.add_argument('--mask_prob', type=float, default=0.3, help='Masking Drop Probability. Only Used for Only Double MHA')
 
    # AMSoftmax Config
    parser.add_argument('--scalingFactor', type=float, default=30.0, help='')
    parser.add_argument('--marginFactor', type=float, default=0.4, help='')
    parser.add_argument('--annealing', action='store_true')

    # Optimization 
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD', 'RMSprop'], default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='number of sequences to train on in parallel')
    parser.add_argument('--gradientAccumulation', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=1000000, help='number of full passes through the trainning data')
    parser.add_argument('--early_stopping', type=int, default=25, help='-1 if not early stopping')
    parser.add_argument('--print_every', type = int, default = 1000)
    parser.add_argument('--requeue',action='store_true', help='restart from the last model for requeue on slurm')
    parser.add_argument('--validate_every', type = int, default = 5000)
    parser.add_argument('--num_workers', type = int, default = 2)
    
    # parse input params
    params=parser.parse_args()
    params.num_spkrs = 6

    main(params)
