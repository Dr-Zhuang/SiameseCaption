import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import json
import logging
import nltk
import os
import pickle
from PIL import Image
from random import random, randint
from time import time

logger = logging.getLogger('caption')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('training.log')
fh.setFormatter(formatter)

COCO_IMAGE_PATH = 'data/train2017/'
COCO_JSON_PATH = 'data/annotations/captions_train2017.json'
MODEL_PATH = 'model/'
VOCAB_PATH = 'data/vocab.pkl'

BATCH_SIZE = 8
EPOCH = 3
LAERNING_RATE = 0.0001
POSITIVE_RATE = 0.65

class Vocabulary(object):

    def __init__(self):

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):

        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):

        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        
        return len(self.word2idx)

class CocoDataset(Dataset):

    def __init__(self, imageFolder, jsonFile, vocabFile, positiveRate, transfrom):

        self.positiveRate = positiveRate
        with open(jsonFile, 'r') as fin:
            self.captions = json.load(fin)['annotations']
        self.folder = imageFolder
        self.transform = transform
        with open(vocabFile, 'rb') as fin:
            self.vocab = pickle.load(fin)

    def __len__(self):

        return len(self.captions)

    def __getitem__(self, index):

        captionRaw = self.captions[index]['caption']
        totalVocab = len(self.vocab)
        tokens = nltk.tokenize.word_tokenize(str(captionRaw).lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        caption = torch.Tensor(caption)

        if random() < POSITIVE_RATE:

            imageFile = '{}.jpg'.format(str(self.captions[index]['image_id']).zfill(12))
            image = Image.open(os.path.join(self.folder, imageFile)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

            return image, caption, 0.0

        else:

            imageFile = os.listdir(self.folder)[randint(0, len(os.listdir(self.folder)) - 1)]
            while imageFile == '{}.jpg'.format(str(self.captions[index]['image_id']).zfill(12)):
                imageFile = os.listdir(self.folder)[randint(0, len(os.listdir(self.folder) - 1))]
            image = Image.open(os.path.join(self.folder, imageFile)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

            return image, caption, 1.0

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, label = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths, label

class EncoderCNN(nn.Module):
    
    def __init__(self, featureSize):
        
       super(EncoderCNN, self).__init__()
       self.baseCNN = models.resnet101()
       self.linear = nn.Linear(1000, featureSize)

    def forward(self, image):

        return self.linear(self.baseCNN(image))

class DecoderLSTM(nn.Module):

    def __init__(self, featureSize, vocabSize):

        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTMCell(featureSize, vocabSize)
        self.embedding = nn.Embedding(vocabSize, featureSize)

    def forward(self, feature, maxLength):

        output = []
        hidden, state = self.lstm(feature)
        softmaxResult = F.softmax(hidden, dim=0)
        _, wordChoice = softmaxResult.max(1)
        output.append(wordChoice)
        for i in range(maxLength - 1):
            newInput = self.embedding(wordChoice)
            hidden, state = self.lstm(newInput, (hidden, state))
            softmaxResult = F.softmax(hidden, dim=0)
            _, wordChoice = softmaxResult.max(1)
            output.append(wordChoice)
        return torch.stack(output)

class SiameseLSTM(nn.Module):
    
    def __init__(self, vocabSize, featureSize):

        super(SiameseLSTM, self).__init__()
        self.lstm = nn.LSTM(featureSize, featureSize)
        self.embedding = nn.Embedding(vocabSize, featureSize)

    def forward(self, caption):
        
        embedded = self.embedding(caption)
        results, (hiddens, cells) = self.lstm(embedded)
        return hiddens[0]

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=4.0):

        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
	                              (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
	
        return loss_contrastive

def oneHot(index, totalVocab):

    temp = [0.0 for i in range(totalVocab)]
    temp[index] = 1.0

    return temp

def prepareCaption(caption, totalVocab):

    captionNew = []
    for i in range(caption.size(0)):
        encodedCaption = []
        for j in range(caption.size(1)):
            encodedCaption.append(oneHot(caption[i, j], totalVocab))
        captionNew.append(encodedCaption)
    
    return torch.transpose(torch.tensor(captionNew), 0, 1).cuda()

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = CocoDataset(COCO_IMAGE_PATH, COCO_JSON_PATH, VOCAB_PATH, POSITIVE_RATE, transfrom=transform)
train_dataloader = DataLoader(trainset, shuffle=True, num_workers=8, batch_size=BATCH_SIZE, collate_fn=collate_fn)

#SiaCNN = models.resnet101().cuda() save GPU
SiaLSTM = SiameseLSTM(len(trainset.vocab), 512).cuda()
#encoder = models.resnet50().cuda()
#SiaCNN = EncoderCNN(512).cuda()
encoder = EncoderCNN(512).cuda()
decoder = DecoderLSTM(512, len(trainset.vocab)).cuda()

#optimizerD = optim.Adam(list(SiaCNN.parameters()) + list(SiaLSTM.parameters()), LAERNING_RATE)
optimizerD = optim.SGD(list(encoder.parameters()) + list(SiaLSTM.parameters()), LAERNING_RATE) # save GPU
optimizerG = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), LAERNING_RATE)
criterion = ContrastiveLoss()

def train(epoch, modelPath):

    logPath = os.path.join(modelPath, 'log.txt')
    totalStep = len(trainset) // BATCH_SIZE
    totalVocab = len(trainset.vocab)
    print('training started, total step = {}'.format(totalStep))

    for _ in range(epoch):
        
        t = time()

        for i, (image, caption, lengths, labels) in enumerate(train_dataloader):

            image, label = image.cuda(), labels
            #SiaCNN.zero_grad()
            encoder.zero_grad() # save GPU
            SiaLSTM.zero_grad()
            #output1 = SiaCNN(image)
            output1 = encoder(image) # save GPU
            #captionIn = prepareCaption(caption, totalVocab)
            captionIn = torch.transpose(caption, 0, 1).cuda()
            output2 = SiaLSTM(captionIn)
            #lossD1 = criterion(output1, output2, label)
            lossD1 = criterion(output1, output2, torch.tensor(label).cuda()) # force to same
            lossD1.backward(retain_graph=True)

            feature = encoder(image)
            text = decoder(feature, 20)
            output3 = SiaLSTM(text)
            #lossD2 = criterion(output1, output3, torch.full((dataSize,), 1).cuda())
            lossD2 = criterion(output1, output3, 1) # force to not same
            lossD2.backward(retain_graph=True)
            optimizerD.step()

            encoder.zero_grad()
            decoder.zero_grad()
            #lossG = criterion(output1, output3, torch.full((dataSize,), 0).cuda())
            lossG = criterion(output1, output3, 0) # force to same
            lossG.backward(retain_graph=True)
            optimizerG.step()

            if 0 == i % 50:
                t = time() - t
                speed = 50 / t
                t = time()
                eta = int((totalStep * (epoch - _) - i) / speed)
                print("[{}/{}][{}/{}]: lossD = {}, lossG = {}\n eta {}:{}:{}"
                        .format(_, epoch, i, totalStep, lossD1.item() + lossD2.item(), lossG.item(),
                                eta // 3600, (eta // 60) % 60, eta % 60))
                with open(logPath, 'a+') as fout:
                    fout.write('[{}/{}][{}/{}]: lossD = {}, lossG = {}\n'
                        .format(_ + 1, epoch, i, totalStep, lossD1.item() + lossD2.item(), lossG.item()))

            if 999 == i % 20000:
                #torch.save(SiaCNN.state_dict(), os.path.join(modelPath, 'SiaCNN-{}-{}.pth'.format(_, i)))
                torch.save(SiaLSTM.state_dict(), os.path.join(modelPath, 'SiaLSTM-{}-{}.pth'.format(_, i)))
                torch.save(encoder.state_dict(), os.path.join(modelPath, 'Siaencoder-{}-{}.pth'.format(_, i)))
                torch.save(decoder.state_dict(), os.path.join(modelPath, 'Siadecoder-{}-{}.pth'.format(_, i)))
                print('{}-{} steps models saved to {}'.format(_, i, modelPath))

    #torch.save(SiaCNN.state_dict(), os.path.join(modelPath, 'SiaCNN-{}-{}.pth'.format(_, i)))
    torch.save(SiaLSTM.state_dict(), os.path.join(modelPath, 'SiaLSTM-{}-{}.pth'.format(_, i)))
    torch.save(encoder.state_dict(), os.path.join(modelPath, 'Siaencoder-{}-{}.pth'.format(_, i)))
    torch.save(decoder.state_dict(), os.path.join(modelPath, 'Siadecoder-{}-{}.pth'.format(_, i)))
                
if __name__ == '__main__':
    train(EPOCH, MODEL_PATH)