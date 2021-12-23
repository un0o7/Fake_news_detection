import pandas as pd 
import numpy as np 
import json, time 
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
import warnings
import re
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.utils.data import Dataset
# 定义model
class Multi_Model(nn.Module):
    def __init__(self, bert_path, classes=2 , p=10):
        super(Multi_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)     # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size,p)  # 直接分类
        '''
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        '''
        #self.image_fc1 = nn.Linear(num_ftrs,  p)
        # input 3*224*224
        self.cnn = nn.Sequential(
            nn.Conv2d( 3, 1,kernel_size=5,stride=2,padding=2),# 1 * 112*112
            nn.ReLU(),
            nn.MaxPool2d(2),#1*56*56
            nn.Conv2d(1,1,kernel_size=5,stride = 2 ,padding = 0),#1*26*26
            nn.ReLU(),
        )
        self.image_fc = nn.Sequential(
            nn.Linear(1*26*26,26),
            nn.Linear(26,p)
        )
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module(
            'c_fc1', nn.Linear(2 *p, p))
        self.class_classifier.add_module('c_fc2',nn.Linear(p,2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,image=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]   # 池化后的输出 [bs, config.hidden_size]
        text = self.fc(out_pool)   #  [bs, classes]
        #image = self.vgg(image)  # [N, 512]
        #image = F.leaky_relu(self.image_fc1(image))
        image = self.cnn(image)
        image = self.image_fc(image.view(image.size(0),-1))
        
        text_image = torch.cat((text, image), 1)
        class_output = self.class_classifier(text_image)
        return class_output

