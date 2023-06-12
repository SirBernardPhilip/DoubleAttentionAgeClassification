import os
import sys
import argparse
import numpy as np
import random
import pickle
import time
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
sys.path.append('./scripts/')
from data import *
from model_softmax import SpeakerClassifier
from loss import *
from utils import *


wandb.init(project="ca&es&en", entity="davilin")
path_net = "/home/usuaris/veu/david.linde/DoubleAttentionSpeakerVerification/scripts/models/ca_es_en_losses_comb_v2/resnet_resnet_1.0_128batchSize_0.0001lr_0.001weightDecay_512kernel_400embSize_30.0s_0.4m_DoubleMHA_32_50000.chkpt"

classes = ('0','1','2','3','4','5','6','7','8')

class Trainer:

    def __init__(self, params, device):

        self.params = params
        self.device = device
        self.__load_network()
        self.__load_data()
        self.__load_optimizer()
        self.__load_criterion()
        self.__initialize_training_variables()

    def __load_network(self):

        self.net = SpeakerClassifier(self.params, self.device)
        print(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        checkpoint = torch.load(path_net)
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.params = checkpoint['settings']
        self.starting_epoch = checkpoint['epoch']+1
        self.step = 0
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = nn.DataParallel(self.net)

    def __load_previous_states(self):

        list_files = os.listdir(self.params.out_dir)
        list_files = [self.params.out_dir + '/' + f for f in list_files if '.chkpt' in f]
        if list_files:
            file2load = max(list_files, key=os.path.getctime)
            checkpoint = torch.load(file2load, map_location=self.device)
            try:
                self.net.load_state_dict(checkpoint['model'])
            except RuntimeError:
                self.net.module.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.params = checkpoint['settings']
            self.starting_epoch = checkpoint['epoch']+1
            self.step = checkpoint['step']
            print('Model "%s" is Loaded for requeue process' % file2load)
        else:
            self.step = 0
            self.starting_epoch = 1

    def __initialize_training_variables(self):

        self.best_f1score = 0.2845
        self.stopping = 0.0




    def __load_data(self):
        print('Loading Data and Labels')
        with open(self.params.train_labels_path, 'r') as data_labels_file:
            train_labels=data_labels_file.readlines()

        with open(self.params.validation_file, 'r') as data_validation_labels_file:
            validation_labels=data_validation_labels_file.readlines()
        
        data_loader_parameters = {'batch_size': self.params.batch_size, 'shuffle': True, 'num_workers': self.params.num_workers}
        #data_loader_parameters_val = {'batch_size': 1, 'shuffle': True, 'num_workers': self.params.num_workers}

        self.training_generator = DataLoader(Dataset(train_labels, self.params), **data_loader_parameters)
        self.validation_generator = DataLoader(Dataset(validation_labels, self.params), **data_loader_parameters)


    def __load_optimizer(self):
        if self.params.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        if self.params.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        if self.params.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)

    def __update_optimizer(self):

        if self.params.optimizer == 'SGD' or self.params.optimizer == 'Adam':
            for paramGroup in self.optimizer.param_groups:
                paramGroup['lr'] *= 0.5
            print('New Learning Rate: {}'.format(paramGroup['lr']))
    
    def __load_criterion(self):
        if self.params.criterion == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()
            
        elif self.params.criterion == "CrossEntropyWeighted":
            self.weights = weights(self.params.train_labels_path)
            print("Weights: {}".format(self.weights))
            self.weights = torch.Tensor(self.weights).cuda()
            self.criterion = nn.CrossEntropyLoss(weight=self.weights)
        elif self.params.criterion == "OrdinalReg":
            self.criterion = OrdinalRegression()
        elif self.params.criterion == "OrdinalRegWeighted":
            self.weights = weights(self.params.train_labels_path)
            print("Weights: {}".format(self.weights))
            self.weights = torch.Tensor(self.weights).cuda()
            self.criterion = OrdinalRegressionWeighted(self.weights)  
        elif self.params.criterion == "OrdinalTunedReg":
            self.criterion = OrdinalRegressionTuned()

        print("Loss function 1: {}".format(self.params.criterion))

        if self.params.criterion2 == "OrdinalTunedReg":
            self.criterion2 = OrdinalRegressionTuned()
            print("Loss function 2: {}".format(self.params.criterion2))

    def __initialize_batch_variables(self):

        self.print_time = time.time()
        self.train_loss = 0.0
        self.train_accuracy = 0.0
        self.train_batch = 0


    def __validate(self):
        validation_accuracy = 0

        with torch.no_grad():
            valid_time = time.time()
            self.net.eval()
            test_batch = 0
            predictions_fscore = []
            labels_fscore = []
            #iterator = iter(self.validation_generator)
            for input, labels in self.validation_generator:
                input, labels = input.float().to(self.device), labels.long().to(self.device)
                prediction, AMPrediction  = self.net(input, label=labels, step=self.step)
                #prediction = self.net(input, label=labels, step=self.step)
                predictions_fscore.append(prediction)
                labels_fscore.append(labels)
                validation_accuracy += Accuracy(prediction, labels)
                test_batch +=1
            validation_accuracy = validation_accuracy * 100/test_batch
            f1score = fscore(torch.cat(predictions_fscore),torch.cat(labels_fscore))
            modified_predictions, modified_labels = modified_target(torch.cat(predictions_fscore),torch.cat(labels_fscore))
            mse = nn.MSELoss(reduction='none')
            error = mse(modified_predictions, modified_labels).sum(axis=1).mean()
            wandb.log({"Fscore validation": f1score})
            wandb.log({"MSE": error})
            print('--Validation Epoch:{epoch: d}, Updates:{Num_Batch: d}, Fscore: {f1score: 3.3f}, Error:{error: 3.3f} ,Validation accuracy:{validation_accuracy: 3.3f}, elapse:{elapse: 3.3f} min'.format(epoch=self.epoch, Num_Batch=self.step, error = error ,f1score = f1score ,validation_accuracy=validation_accuracy, elapse=(time.time()-valid_time)/60))
            # early stopping and save the best model
            if f1score > self.best_f1score:
                self.best_f1score = f1score
                self.stopping = 0
                print('We found a better model!')
                chkptsave(params, self.net, self.optimizer, self.epoch, self.step)

            else:
                self.stopping += 1
                print('Better Accuracy is: {}. {} epochs of no improvement'.format(self.best_f1score, self.stopping))
                self.params.dropout = 0.25
            self.print_time = time.time()
            self.net.train()

    def __update(self):

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1

        if self.step % int(self.params.print_every) == 0:
            wandb.log({"Train accuracy": self.train_accuracy *100/ self.train_batch, "Train loss":self.train_loss / self.train_batch})
            print('Training Epoch:{epoch: d}, Updates:{Num_Batch: d} -----> xent:{xnet: .3f}, Accuracy:{acc: .2f}, elapse:{elapse: 3.3f} min'.format(epoch=self.epoch, Num_Batch=self.step, xnet=self.train_loss / self.train_batch, acc=self.train_accuracy *100/ self.train_batch, elapse=(time.time()-self.print_time)/60))
            self.__initialize_batch_variables()

        # validation
        if self.step % self.params.validate_every == 0:
            self.__validate()

    def __updateTrainningVariables(self):

        if (self.stopping+1)% 15 ==0:
            self.__update_optimizer()

    def __randomSlice(self, inputTensor):
        index = random.randrange(200,self.params.window_size*100)
        return inputTensor[:,:index,:]

    def train(self):

        print('Start Training')
        for self.epoch in range(self.starting_epoch, self.params.max_epochs):  # loop over the dataset multiple times
            self.net.train()
            self.__initialize_batch_variables()
            for input, label in self.training_generator:
                input, label = input.float().to(self.device), label.long().to(self.device)
                input = self.__randomSlice(input) if self.params.randomSlicing else input
                prediction, AMPrediction  = self.net(input, label=label, step=self.step)
                #prediction = self.net(input, label=label, step=self.step)

                loss = self.criterion(AMPrediction, label)

                loss.backward()
                self.train_accuracy += Accuracy(prediction, label)
                self.train_loss += loss.item()
                
                self.train_batch += 1
                if self.train_batch % self.params.gradientAccumulation == 0:
                    self.__update()

            if self.stopping > self.params.early_stopping:
                print('--Best Model Fscore%%: %.2f' %(self.best_f1score))
                break
            
            self.__updateTrainningVariables()


        print('Finished Training')

def main(opt):

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    print('Defining Device')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    print('Loading Trainer')
    trainer = Trainer(opt, device)
    trainer.train()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a VGG based Speaker Embedding Extractor')
   
    parser.add_argument('--train_data_dir', type=str, default='', help='data directory.')
    parser.add_argument('--valid_data_dir', type=str, default='', help='data directory.')
    parser.add_argument('--train_labels_path', type = str, default = 'labels/Vox2.ndx')
    parser.add_argument('--data_mode', type = str, default = 'normal', choices=['normal','window'])
    parser.add_argument('--validation_file', type = str, default='labels/validation.lst')
    parser.add_argument('--out_dir', type=str, default='./models/resnet', help='directory where data is saved')
    parser.add_argument('--model_name', type=str, default='CNN', help='Model associated to the model builded')
    parser.add_argument('--front_end', type=str, default='resnet', choices = ['VGG3L','VGG4L','resnet'], help='Kind of Front-end Used')    
    # Network Parameteres
    parser.add_argument('--window_size', type=float, default=3.5, help='number of seconds per window')
    parser.add_argument('--randomSlicing',action='store_true')
    parser.add_argument('--normalization', type=str, default='cmn', choices=['cmn', 'cmvn'])
    parser.add_argument('--kernel_size', type=int, default=1024)
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
    parser.add_argument('--max_epochs', type=int, default=250, help='number of full passes through the trainning data')
    parser.add_argument('--early_stopping', type=int, default=10, help='-1 if not early stopping')
    parser.add_argument('--print_every', type = int, default = 1000)
    parser.add_argument('--requeue',action='store_true', help='restart from the last model for requeue on slurm')
    parser.add_argument('--validate_every', type = int, default = 5000)
    parser.add_argument('--num_workers', type = int, default = 2)
    parser.add_argument('--criterion', type=str, default='CrossEntropyWeighted',choices=['CrossEntropy','CrossEntropyWeighted','OrdinalReg','OrdinalRegWeighted','OrdinalTunedReg'] ,help='Use of weighted loss (weights imported from dicts.py, they are optimized weighted, no_weighted, weighted_train')
    parser.add_argument('--criterion2', type=str, default='OrdinalTunedReg',choices=['CrossEntropy','CrossEntropyWeighted','OrdinalReg','OrdinalRegWeighted','OrdinalTunedReg'] ,help='Use of weighted loss (weights imported from dicts.py, they are optimized weighted, no_weighted, weighted_train')
    parser.add_argument('--dropout', type = float, default = 0)
    
    # parse input params
    params=parser.parse_args()
    params.model_name = getModelName(params)
    params.num_spkrs = getNumberOfSpeakers(params.train_labels_path) 
    print('{} Range of ages'.format(params.num_spkrs))

    if not os.path.exists(params.out_dir):
        os.makedirs(params.out_dir)

    with open(params.out_dir + '/' + params.model_name + '_config.pkl', 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    main(params)
