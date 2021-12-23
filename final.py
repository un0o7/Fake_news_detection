from model import Multi_Model
import pickle 
from PIL import Image
import re
import os 
from torchvision.transforms.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import warnings
from torch import LongTensor
import torch 
import time 
from sklearn.metrics import accuracy_score, classification_report,recall_score
warnings.filterwarnings('ignore')
class Config():
    def __init__(self):
        self.batch_size = 64
        self.epochs = 5
        self.bert_path = "./bert_model/"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FakeNewsDataset(Dataset):
    def __init__(self,input_three,event,image,label) :
        self.event = LongTensor(list(event)) 
        self.image = torch.FloatTensor([np.array(i) for i in image]) 
        self.label = LongTensor(list(label))
        self.input_three = list()
        self.input_three.append( LongTensor(input_three[0]))
        self.input_three.append(LongTensor(input_three[1]))
        self.input_three.append(LongTensor(input_three[2]))
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        return self.input_three[0][idx],self.input_three[1][idx],self.input_three[2][idx],self.image[idx],self.event[idx],self.label[idx]
def readImage(filepath_list):
    """读取所有图片"""
    #if os.path.exists('./pickles/'):
    #    return pickle.load(open('./pickles/images.pkl','rb'))
    image_list = dict()
    data_transforms = Compose(transforms=[
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for path in list(filepath_list):
        for i, filename in enumerate(os.listdir(path)):
            try:
                im = Image.open(os.path.join(path, filename)).convert('RGB')
                im = data_transforms(im)
                image_list[filename.split('/')[-1].split('.')[0].lower()] = im
            except Exception as e:
                print("[-] Error {} when handling {}".format(e, filename))
    # print("[+] Length of `image_list`: {}".format(len(image_list)))

    pickle.dump(image_list,open("./pickles/images.pkl","wb"))
    print("dump successful")
    return image_list
def cleanSST(string):
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()
def load_id(path):
    result = {}
    for p in path:
        id = pickle.load(open(p,'rb'))
        print(len(id))
        result = dict(id,**result)
    print(len(result))
    return result      
def load_text_image():
    result_path ='./pickles/four_data.pkl'
    if os.path.exists('./pickles/'):
        return pickle.load(open(result_path,'rb'))
    text = []
    image = []
    event = []
    label = []
    
    text_list = ['./weibo/tweets/test_nonrumor.txt','./weibo/tweets/test_rumor.txt',
             './weibo/tweets/train_nonrumor.txt','./weibo/tweets/train_rumor.txt']  
    image_list = ['./weibo/rumor_images/','./weibo/nonrumor_images/']
    post_id_path = ['./weibo/train_id.pickle','./weibo/test_id.pickle','./weibo/validate_id.pickle']
    all_id = load_id(post_id_path)
    #all_images = readImage(image_list) #{id:image}
    
    #print('read images ok')

    for index,filename in enumerate(text_list):
        print(index,'  text read ok')
        with open(filename, 'r',encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)): 
                #event : load post id and transform it to event
                if i %3 == 0 :
                    post_id = str(line).split('|')[0]
                    if post_id in all_id.keys():
                        event.append(all_id[post_id])
                        allowed = True  
                    else:
                        allowed = False
                #image : only capture the first image which is in weibo dataset 
                if i %3 == 1  and allowed:
                    image.append(line.strip())
                    '''for image_id in line.strip().split('|'):
                        image_id = image_id.split("/")[-1].split(".")[0]
                        if image_id in all_images:
                            image.append(all_images[image_id])
                            break'''
                #text  : just remove special chacters and strip,do not tokenize 
                if i%3 ==2 and allowed:
                    text.append(cleanSST(line))
                    #label
                    if (index+1) % 2 == 1:  # non-rumor -> 0
                        y = 0
                    else:              # rumor -> 1
                        y = 1
                    label.append(y)
                    allowed = False
    print('read all features ok')
    print(len(text),len(event),len(image),len(label))
    pickle.dump([text,image , event ,label ],open(result_path,'wb'))
    return text , image,event ,label 
def imageurl_image(images):
    all_images = pickle.load(open('./pickles/images.pkl','rb'))
    image = []
    mask = []
    for line in images :
        flag = 0
        for image_id in line.strip().split('|'):
            #print(image_id)
            image_id = image_id.split("/")[-1].split(".")[0]
            #print(image_id)
            if image_id in all_images:
                image.append(all_images[image_id])
                flag = 1 
                break
        mask.append(flag)
        #else:
            #print(line,image_id,'not find') 
    print("preprocess images pk :",len(image))
    return mask,image
def filter(l,mask):
    result = []
    for index,i in enumerate(l):
        if mask[index]:
            result.append(i)
    return result 
def token_text(text):
    max_len = 50
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')   # 初始化分词器
    input_ids, input_masks, input_types,  = [], [], []
    for line in text :
        encode_dict = tokenizer.encode_plus(text=line, max_length=max_len, 
                                                padding='max_length', truncation=True)
        #dec:https://www.cnblogs.com/douzujun/p/13572694.html
        input_ids.append(encode_dict['input_ids'])
        input_types.append(encode_dict['token_type_ids'])
        input_masks.append(encode_dict['attention_mask'])
    return [input_ids,input_types,input_masks]
def get_parameter_number(model):
    #  打印模型参数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)
def evaluate(multi_model,vali_dataloader,device):
    multi_model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for index,(batch_text0,batch_text1,batch_text2,batch_image,batch_event,batch_label) in enumerate(vali_dataloader):
            batch_text0 = batch_text0.to(device)
            batch_text1 = batch_text1.to(device)
            batch_text2 = batch_text2.to(device)
            batch_image = batch_image.to(device)
            batch_event = batch_event.to(device)
            batch_label = batch_label.to(device)
            y_pred = multi_model(batch_text0,batch_text1,batch_text2,batch_image)
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(batch_label.squeeze().cpu().numpy().tolist())
    
    return accuracy_score(val_true, val_pred)  #返回accuracy
def predict():
    config = Config()
    data_transforms = Compose(transforms=[
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text = input("fake news:")
    image_path = input("image path:")
    multi_model = Multi_Model(config.bert_path)
    multi_model.load_state_dict(torch.load('best_multi_bert_model.pth'))
    im = Image.open(image_path).convert('RGB')
    im = data_transforms(im)
    # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') 
    input_ids, input_masks, input_types,  = [], [], []
    encode_dict = tokenizer.encode_plus(text=cleanSST(text), max_length=50, 
                                                padding='max_length', truncation=True)
    input_ids.append(encode_dict['input_ids'])
    input_types.append(encode_dict['token_type_ids'])
    input_masks.append(encode_dict['attention_mask'])
    label_pred = multi_model(LongTensor(input_ids),LongTensor(input_types),LongTensor(input_masks),[np.array(im)])
    y_pred = torch.argmax(label_pred, dim=1).detach().cpu().numpy().tolist()    
    print(y_pred)
train_acc_vector = []
vali_acc_vector = []
def train_val_test():
    #train , test , validate 
    config = Config()
    text , image,event ,label = pickle.load(open("./pickles/all_data.pkl",'rb'))
    train_len = int(0.6 * len(label))
    test_len = int(0.2*len(label) )
    validate_len = len(label) - train_len - test_len
    print("train:{} test:{} validate:{}".format(train_len,test_len,validate_len))
    print("fake news:{}  Real news:{}".format(sum(label),len(label)-sum(label)))
    text = np.array(text)
    image = np.array(image)
    event = np.array(event)
    label = np.array(label)
    
    idxes = np.arange(len(label))
    np.random.seed(2019)   # 固定种子
    np.random.shuffle(idxes)
    print("load dataset")
    train_dataset = FakeNewsDataset(np.array([text[0][idxes[:train_len]],text[1][idxes[:train_len]],text[2][idxes[:train_len]]]),
                                    event[idxes[:train_len]],image[[idxes[:train_len]]],label[idxes[:train_len]])
    pickle.dump(train_dataset,open('./pickles/train_dataset.pkl','wb'))
    test_dataset = FakeNewsDataset(np.array([text[0][idxes[train_len:test_len+train_len]],text[1][idxes[train_len:test_len+train_len]],text[2][idxes[train_len:test_len+train_len]]]),
                                    event[idxes[train_len:test_len+train_len]],image[[idxes[train_len:test_len+train_len]]],label[idxes[train_len:test_len+train_len]])
    pickle.dump(test_dataset,open('./pickles/test_dataset.pkl','wb'))
    validate_dataset = FakeNewsDataset(np.array([text[0][idxes[test_len+train_len:]],text[1][idxes[test_len+train_len:]],text[2][idxes[test_len+train_len:]]]),
                                    event[idxes[test_len+train_len:]],image[[idxes[test_len+train_len:]]],label[idxes[test_len+train_len:]]) 
    pickle.dump(validate_dataset,open('./pickles/validate_dataset.pkl','wb'))
    print("dataset dump ok")

    train_dataset = pickle.load(open('./pickles/train_dataset.pkl','rb'))
    test_dataset = pickle.load(open('./pickles/test_dataset.pkl','rb'))
    validate_dataset = pickle.load(open('./pickles/validate_dataset.pkl','rb'))
    print(len(train_dataset),len(test_dataset),len(validate_dataset))
    print("dataset dump ok")
    train_loader = DataLoader(train_dataset,batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset,batch_size=config.batch_size)
    validate_loader = DataLoader(validate_dataset,batch_size=config.batch_size)
    print('process data  Loader success')
    bert_multi_model = Multi_Model(config.bert_path)
    print("model init ok")
    print(get_parameter_number(bert_multi_model))
    optimizer = AdamW(bert_multi_model.parameters(), lr=2e-5, weight_decay=1e-4) #AdamW优化器
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                            num_training_steps=config.epochs*len(train_loader))
    criterion = nn.CrossEntropyLoss()
    
    
    #official train
    best_acc = 0.0
    for i in range(config.epochs):
        start = time.time()
        bert_multi_model.train()
        print("***** Running training epoch {} *****".format(i+1))
        train_loss_sum = 0.0
        label_predict = []
        label_epoch = []
        for index,(batch_text0,batch_text1,batch_text2,batch_image,batch_event,batch_label) in enumerate(train_loader):
            batch_text0 = Variable(batch_text0.to(config.device))
            batch_text1 = Variable(batch_text1.to(config.device))
            batch_text2 = Variable(batch_text2.to(config.device))
            batch_image = Variable(batch_image.to(config.device))
            batch_event = Variable(batch_event.to(config.device))
            batch_label = Variable(batch_label.to(config.device))
            #y_pred = model(ids, att, tpe)
            print("training")
            label_pred,domain_pred = bert_multi_model(batch_text0,batch_text1,batch_text2,batch_image)
            loss = criterion(label_pred,batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()   # 学习率变化
            label_predict.extend(torch.argmax(label_pred, dim=1))
            label_epoch.extend(batch_label)
            train_loss_sum += loss.item()
            if (index + 1) % (len(train_loader)//5) == 0:    # 只打印五次结果
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                          i+1, index+1, len(train_loader), train_loss_sum/(index+1), time.time() - start))
        epoch_acc = accuracy_score(label_predict,batch_label)
        train_acc_vector.append(epoch_acc)
        print("Train Accuracy:{} Recall:{}".format(epoch_acc,recall_score(label_predict,batch_label)))
        
        bert_multi_model.eval()
        acc = evaluate(bert_multi_model, validate_loader, config.device)  # 验证模型的性能
        vali_acc_vector.append(acc)
        ## 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(bert_multi_model.state_dict(), "best_multi_bert_model.pth") 
        
        print("current acc is {:.4f}, best acc is {:.4f}".format(acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))
    test_model = Multi_Model(config.bert_path)
    test_model.load_state_dict(torch.load("best_multi_bert_model.pth"))
    evaluate(test_model,test_loader,config.device)
              
def main():
    train_val_test()

if __name__ == '__main__':
    '''
    text , image,event ,label = pickle.load(open('./pickles/all_data.pkl','rb'))
    map_id = {}
    for  index,event_num in enumerate(event) :
        if event_num in map_id:
            
            event[index] = map_id[event_num]
            
        else:
            map_id[event_num] = len(map_id)
            event[index] = map_id[event_num]
    print(event[:100])
    print("total event number",len(map_id))
    '''
            
    #mask , image = imageurl_image(image)
    #text = filter(text,mask)
    #event = filter(event,mask)
    #label = filter(label,mask) 
    #text =  token_text(text)
    #print(len(text),len(image),len(event),len(label))
    #pickle.dump([text,image , event ,label ],open('./pickles/all_data.pkl','wb'))
    main()
    
    