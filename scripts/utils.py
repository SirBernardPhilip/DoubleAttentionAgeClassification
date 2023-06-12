import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def scoreCosineDistance(emb1, emb2):

    dist = F.cosine_similarity(emb1,emb2, dim=-1, eps=1e-08)
    return dist

def chkptsave(opt,model,optimizer,epoch,step):
    ''' function to save the model and optimizer parameters '''
    if torch.cuda.device_count() > 1:
        checkpoint = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'settings': opt,
            'epoch': epoch,
            'step':step}
    else:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'settings': opt,
            'epoch': epoch,
            'step':step}

    torch.save(checkpoint,'{}/{}_{}.chkpt'.format(opt.out_dir, opt.model_name,step))

def Accuracy(pred, labels):

    acc = 0.0
    num_pred = pred.size()[0]
    pred = torch.max(pred, 1)[1]
    for idx in range(num_pred):
        if pred[idx].item() == labels[idx].item():
            acc += 1

    return acc/num_pred

def AccuracyLabel(pred,labels):
    acc = 0.0
    num_pred = pred.size()[0]
    for idx in range(num_pred):
        if pred[idx].item() == labels[idx].item():
            acc += 1

    return acc/num_pred  

def prediction2prediction(pred):
    """Convert ordinal predictions to class labels, e.g.
    
    [0.9, 0.1, 0.1, 0.1] -> [1, 0, 0, 0]
    [0.9, 0.9, 0.1, 0.1] -> [0, 1, 0, 0]
    [0.9, 0.9, 0.9, 0.1] -> [0, 0, 1, 0]
    etc.
    """ 
    prediction = prediction2label(pred=pred.detach().cpu().numpy())
    print(prediction)
    return F.one_hot(prediction,num_classes=6)

def cdw(predictions: [[float]], targets: [float]):

    modified_predictions = torch.zeros_like(predictions)
    for i, target in enumerate(targets):
        for j in range(0,6):
            modified_predictions[i,j] = predictions[i,j]*abs(target-j+1)
                
    return  modified_predictions.to(dtype=torch.float32)

def prediction2label(pred: np.ndarray):
    """Convert ordinal predictions to class labels, e.g.
    
    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    print(pred)
    return torch.from_numpy((pred > 0.5).cumprod(axis=1).sum(axis=1) - 1)

def getNumberOfSpeakers(labelsFilePath):

    speakersDict = dict()
    with open(labelsFilePath,'r') as labelsFile:
        for line in labelsFile.readlines():
            speakersDict[line.split()[1]] = 0
    return len(speakersDict)

def getModelName(params):

    model_name = params.model_name

    model_name = model_name + '_{}'.format(params.front_end) + '_{}'.format(params.window_size) + '_{}batchSize'.format(params.batch_size*params.gradientAccumulation) + '_{}lr'.format(params.learning_rate) + '_{}weightDecay'.format(params.weight_decay) + '_{}kernel'.format(params.kernel_size) +'_{}embSize'.format(params.embedding_size) + '_{}s'.format(params.scalingFactor) + '_{}m'.format(params.marginFactor)

    model_name += '_{}'.format(params.pooling_method) + '_{}'.format(params.heads_number)

    return model_name

def fscore(prediction, labels):
    pred = torch.max(prediction, 1)[1].cpu().numpy()
    labels = labels.cpu()
    return f1_score(labels, pred, average ='macro')

def fscore_modified(prediction, labels):
    preds = torch.max(prediction, 1)[1].cpu().numpy()
    labels = labels.cpu()
    i = 0
    for pred, label in zip(preds,labels):
        if pred == label-1 or pred == label+1:
            preds[i]=label
        i=i+1

    return f1_score(labels, preds, average ='macro')


def weights(path):    
    data = pd.read_csv(path, sep = ' ', header=None)
    total_values = data.shape[0]
    values = data.iloc[:,4].value_counts(sort = False)
    normedWeights = [1 - (x / total_values) for x in values]
    return normedWeights

def modified_target(predictions, labels):

    predictions = torch.max(predictions, 1)[1].cpu().numpy()
    labels = labels.cpu()
    modified_predictions = torch.zeros(predictions.size,6)
    modified_labels = torch.zeros(predictions.size,6)
        # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, prediction in enumerate(predictions):
        #modified_predictions[0:prediction+1] = 1
        modified_predictions[i,0:prediction+1] = 1
    
    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, label in enumerate(labels):
        #modified_labels[0:label+1] = 1
        modified_labels[i, 0:label+1] = 1
    
    
    return modified_predictions.cpu(), modified_labels.cpu()

def modified_label(labels):
    modified_labels = torch.zeros(labels.size(dim=0),6)

    for i, label in enumerate(labels):
        #modified_labels[0:label+1] = 1
        modified_labels[i, 0:label+1] = 1
    
    return modified_labels.cuda().long()

def pca(prediction, target, name):
    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()
    target_l, target_c = df2list(target)
    target = np.array(target_l)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(prediction)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3' ])
    targetDf = pd.DataFrame(data = target, columns = ['target'])
    
    finalDf = pd.concat([principalDf, targetDf], axis = 1)
    finalDf.to_csv(f"/home/usuaris/veu/david.linde/CommonVoice11/{name}_.tsv", sep='\t')

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    #ax.set_zlabel('Principal Component 3', fontsize = 15)


    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['twenties','thirties','fourties','fifties','sixties','seventies']
    colors = ['red', 'green', 'blue','orange','purple','pink']
    #targets, targets_c = df2list(target)
    x = finalDf['principal component 1'].to_list()
    y = finalDf['principal component 2'].to_list()
    #z = finalDf['principal component 3'].to_list()
    #ax.scatter(x, y, z, c=targets_c, label=labels)
    
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color)
    ax.legend(targets)
    ax.grid()

    
    #ax.legend(labels=['twenties','thirties','fourties','fifties','sixties','seventies'])
    #ax.grid()
    ax.figure.savefig(f"/home/usuaris/veu/david.linde/CommonVoice11/{name}_pca.png")

def tsne(prediction, target, name):

    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()
    target_l, target_c = df2list(target)
    target = np.array(target_l)
    tsne = TSNE(n_components = 3)
    tsne_data = tsne.fit_transform(prediction)

    principalDf = pd.DataFrame(data = tsne_data, columns = ['tsne 1', 'tsne 2','tsne 3'])
    targetDf = pd.DataFrame(data = target, columns = ['target'])
    
    finalDf = pd.concat([principalDf, targetDf], axis = 1)
    finalDf.to_csv(f"/home/usuaris/veu/david.linde/CommonVoice11/{name}_tsne.tsv", sep='\t')

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('TSNE 1', fontsize = 15)
    ax.set_ylabel('TSNE 2', fontsize = 15)
    #ax.set_zlabel('Principal Component 3', fontsize = 15)


    ax.set_title('2 component TSNE', fontsize = 20)
    targets = ['twenties','thirties','fourties','fifties','sixties','seventies']
    colors = ['red', 'green', 'blue','orange','purple','pink']
    #targets, targets_c = df2list(target)
    x = finalDf['tsne 1'].to_list()
    y = finalDf['tsne 2'].to_list()
    #z = finalDf['principal component 3'].to_list()
    #ax.scatter(x, y, z, c=targets_c, label=labels)
    
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'tsne 1']
               , finalDf.loc[indicesToKeep, 'tsne 2']
               , c = color)
    ax.legend(targets)
    ax.grid()
    ax.figure.savefig(f"/home/usuaris/veu/david.linde/CommonVoice11/{name}_tsne.png")


def centroid(prediction, target, name):

    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()
    target_l, target_c = df2list(target)
    target = np.array(target_l)
    model_centroid = NearestCentroid(metric='cosine')
    model_centroid.fit(prediction,target)
    centroids = model_centroid.centroids_
    print(centroids)
    centroids = np.array(centroids)
    cosine_distances = cosine_similarity(centroids)
    print(cosine_distances)
    sns.heatmap(cosine_distances, annot=True, cmap="YlGnBu")
    plt.savefig(f"/home/usuaris/veu/david.linde/CommonVoice11/{name}_centroid.png")
    



def df2list(targets):
    targets_c = []
    targets_l = []
    for i in range(0,targets.size):

        if targets[i] == 0:
            targets_l.append("twenties")
            targets_c.append("red")
           
        if targets[i] == 1:
            targets_l.append("thirties")
            targets_c.append("green")

        if targets[i] == 2:
            targets_l.append("fourties")
            targets_c.append("blue")


        if targets[i] == 3:
            targets_l.append("fifties")
            targets_c.append("orange")


        if targets[i] == 4:
            targets_l.append("sixties")
            targets_c.append("purple")


        if targets[i] == 5:
            targets_l.append("seventies")
            targets_c.append("pink")

    return targets_l, targets_c
