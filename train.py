
'''
https://github.com/miraclewkf/ImageClassification-PyTorch/blob/master/level2/train_customData.py
the prosess of the training
训练的整个过程。
可以在train各种的模型，只要把模型搭建好。
这个就是公用的用于训练的文件。
'''

import torch
from torch.autograd import Variable
import os
import time


#构建了整个的训练过程。
def train_model(model, criterion, optimizer, scheduler, num_epoches, use_gpu, dataloaders, batch_size, dataset_sizes):
    since = time.time()  #放置一个时间节点，开始训练的时间
    best_model_wts = model.state_dict()  #保存模型的参数
    best_acc = 0.0

#训练中的epoch的过程
    for epoch in range(num_epoches):
        begin_time = time.time()
        count_batch = 0                                         #batch的数
        print('Epoch {}/{}'.format(epoch, num_epoches-1))
        print('-'*10)

        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()    #这个是啥
                model.train(True)  #训练模式
            else:
                model.train(False) #评价模式


            #初始化loss和正确率
            running_loss = 0.0
            running_corrects = 0

            #从对应的数据集中提取数据
            for data in dataloaders[phase]:
                #每次提取一次数据就是一个batch。训练batch_size个数据

                count_batch += 1

                #在使用过程中，就是dataloader中获得图像和标签
                #dataloader 的输出是tensor    获取数据并转化为变量
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs,labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                _,preds = torch.max(outputs.data,1)
                loss = criterion(outputs,labels)

                if phase =='train':
                    loss.backward()
                    optimizer.step()

                running_loss  +=loss.data[0]
                running_corrects += torch.sum(preds == labels.data).data[0]

                if count_batch%10 ==0:
                    batch_loss = running_loss / (batch_size * count_batch)
                    batch_acc = running_corrects / (batch_size *count_batch)
                    print('{} Epoch [{}] Batch[{}] Loss:{:.4f} Acc: {:.4f} Time: {:.4f}s'.\
                          format(phase, epoch, count_batch, batch_loss, batch_acc, time.time()-begin_time))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))

        #save model
            if phase == 'train':
                if not os.path.exists('output'):
                    os.makedirs('output')
                torch.save(model,'output/resnet_epoch{}.pkl'.format(epoch))

            if phase == 'val' and epoch_acc> best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()    #模型的参数保存

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed //60, time_elapsed %60))
    print('Bet val Acc: {:4f}'.format(best_acc))

        # load best model weights
    model.load_state_dict(best_model_wts)
    return model









































