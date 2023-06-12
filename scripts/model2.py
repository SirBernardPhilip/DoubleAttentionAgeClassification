import torch
from torch import nn
from torch.nn import functional as F
from poolings import *
from CNNs import *
from loss import *
from resnet import *
from collections import namedtuple

class SpeakerClassifier(nn.Module):

    def __init__(self, parameters, device):
        super().__init__()
       
        parameters.feature_size = 80
        self.dropout = parameters.dropout 
        self.device = device
        self.__initFrontEnd(parameters)        
        self.__initPoolingLayers(parameters)
        self.__initFullyConnectedBlock(parameters)
        self.predictionLayer = AMSoftmax(parameters.embedding_size, parameters.num_spkrs, s=parameters.scalingFactor, m=parameters.marginFactor, annealing = parameters.annealing)
        self.softmax = nn.Softmax(dim=1)

    def __initFrontEnd(self, parameters):

        if parameters.front_end=='VGG3L':
            self.vector_size = getVGG3LOutputDimension(parameters.feature_size, outputChannel=parameters.kernel_size)
            self.front_end = VGG3L(parameters.kernel_size)
        
        if parameters.front_end=='VGG4L':
            self.vector_size = getVGG4LOutputDimension(parameters.feature_size, outputChannel=parameters.kernel_size)
            self.front_end = VGG4L(parameters.kernel_size)

        if parameters.front_end=='resnet':

            ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
            resnet18_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [2,2,2],
                               channels = [int(parameters.kernel_size/8), int(parameters.kernel_size/4), int(parameters.kernel_size/2)])
            self.vector_size = int(parameters.kernel_size/2) #getVGG4LOutputDimension(parameters.feature_size, outputChannel=parameters.kernel_size)
            print("Vector_size: ", self.vector_size)
            self.front_end = ResNet(resnet18_config, self.vector_size)


    def __initPoolingLayers(self,parameters):    

        self.pooling_method = parameters.pooling_method

        if self.pooling_method == 'Attention':
            self.poolingLayer = Attention(self.vector_size)
        elif self.pooling_method == 'MHA':
            self.poolingLayer = MultiHeadAttention(self.vector_size, parameters.heads_number)
        elif self.pooling_method == 'DoubleMHA':
            self.poolingLayer = DoubleMHA(self.vector_size, parameters.heads_number, mask_prob = parameters.mask_prob)
            self.vector_size = self.vector_size//parameters.heads_number

    def __initFullyConnectedBlock(self, parameters):

        self.fc1 = nn.Linear(self.vector_size, parameters.embedding_size)
        self.b1 = nn.BatchNorm1d(parameters.embedding_size)
        self.fc2 = nn.Linear(parameters.embedding_size, parameters.embedding_size)
        self.b2 = nn.BatchNorm1d(parameters.embedding_size)
        self.preLayer = nn.Linear(parameters.embedding_size, parameters.embedding_size)
        self.b3 = nn.BatchNorm1d(parameters.embedding_size)

        
    def getEmbedding(self,x):

        encoder_output = self.front_end(x)
        embedding0, alignment = self.poolingLayer(encoder_output)
        embedding1 = F.relu(self.fc1(embedding0))
        embedding2 = self.b2(F.relu(self.fc2(embedding1)))
    
        return embedding2 

    def forward(self, x, label=None, step=0):

        encoder_output = self.front_end(x)
        embedding0, alignment = self.poolingLayer(encoder_output)

        if self.dropout > 0:
            self.drop = nn.Dropout(p=self.dropout)
            embedding1 = self.drop(F.relu(self.fc1(embedding0)))
            embedding2 = self.drop(self.b2(F.relu(self.fc2(embedding1))))
        else:
            embedding1 = F.relu(self.fc1(embedding0))
            embedding2 = self.b2(F.relu(self.fc2(embedding1)))
        embedding3 = self.preLayer(embedding2)
        
        prediction, ouputTensor = self.predictionLayer(embedding3, label, step)
        
        #return self.softmax(embedding4)
        return prediction, ouputTensor

