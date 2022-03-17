# fake news detection through improved EANN with BERT

This task is for my datamining course project. And we use the weibo dataset which is from EANN. 

The result show that our work achieve approximately **91% accuracy **while  original EANN only achieve 82% and in our experiment it only achieve 67 instead of 82% which is claimed in its paper.

You can find our work on kaggle [fakenews-bert | Kaggle](https://www.kaggle.com/hjhsdsdww/fakenews-bert)

Original EANN： https://doi.org/10.1145/3219819.3219903   code: https://github.com/yaqingwang/EANN-KDD18

## what we do 

First, have a look at original EANN model

![image-20211223181011729](https://img2020.cnblogs.com/blog/2348945/202112/2348945-20211223181014822-964823199.png)

It uses Text-CNN as the text feature extractor which is poor in unstanding the semantics of sentences. 

**So we choose to use Bert as the text feature extractor.** 

Because of the fact that there are too many parameters if we use Bert and VGG-19 the same time.

They are both pretraining model. So we just use **two layer CNN** as the image feature extractor.

## how to do 

First, we used Bert to finish a  simple text classification and got  good results compared to other models.(**bert.py**)

And we also known how to use Bert model of transformers framework.

And then we try to extract features from original dataset. For the reason that there are too many data we use many pickle files to save features. And in kaggle, it seems that its environment does not use virtual memmory. And if there are too mant data in CPU/GPU memory, the environment will be interrupted.

The multi model we build is in **model.py**.

## result 

![image-20211223175631716](https://img2020.cnblogs.com/blog/2348945/202112/2348945-20211223175634510-1965998019.png)

![image-20211223175641680](https://img2020.cnblogs.com/blog/2348945/202112/2348945-20211223175642401-686724275.png)



**If you have any questions, please inform me.** 

