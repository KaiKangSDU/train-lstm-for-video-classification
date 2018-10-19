import torch.nn as nn
import torch
from VGG_Face_torch import VGG_Face_torch
from load_image import customData
from torchvision import transforms
from torch.autograd import Variable
import os
import numpy, scipy.io


#搭建了vgg face 模型
'''
采用vggface预训练模型，提取face的特征，保存为mat文件，为之后的LSTM的输入做准备

load mat file

import scipy.io as scio
file = r'G:\kk_file\EmotiW\AFEW_IMAGE_align_crop_feature\Data\Disgust\010942127\031.mat'
data = scio.loadmat(file)
data['feature']
print(data['feature'].shape)

'''




class VGG_Net(nn.Module):
    def __init__(self, model):
        super(VGG_Net, self).__init__()
        self.pre_model = nn.Sequential(*list(model.children())[:-2])  #删除最后一层
        # self.dropout = nn.Dropout(p=0.8)
        #self.classifier = nn.Linear(4096, 7)

    def forward(self, x):
        x = self.pre_model(x)
        # x = self.dropout(x)

        #存储特征不需要最后一个分类层，最后的结果是 4096维的
        #x = self.classifier(x)

        return x
if __name__ == '__main__':

    data_transforms = {'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        }


    data_path = "G:\kk_file\EmotiW\AFEW_IMAGE_align_crop\Data" #................................................是要改的
    dir = "G:\kk_file\EmotiW\AFEW_IMAGE_align_crop_feature\Data"
    use_gpu = torch.cuda.is_available()
    batch_size = 32
    num_class = 7

        #convert image into List
    image_datasets = {x: customData(img_path=os.path.join(data_path,x),
                                        txt_path=os.path.join(data_path,x)+'\content.txt',
                                        data_transforms=data_transforms,
                                       dataset=x) for x in ['train', 'val']}

        #convert to tensor, 作为模型可以接受的数据，就定义好是不是打乱， 每个batch_size是多少
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=4)
                       for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


    # feature model
    model_emotion = VGG_Face_torch
    model = VGG_Net(model_emotion)
    if torch.cuda.is_available():
        model = VGG_Net(model_emotion).cuda()
    else:
        model = VGG_Net(model_emotion)

    #print(model)


    # load parameters for feature models
    pretrained_dict = torch.load("best_vggface.pkl", map_location='cpu')

    model_dict = model.state_dict()
    pretrained_dict = { k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)


    for phase in ['train','val']:
        for data in dataloaders[phase]:
            inputs, labels, path = data
            inputs, labels = Variable(inputs), Variable(labels)
            out = model(inputs)

            feature_map = out.detach().numpy()       #convert variable into numpy
            #print(feature_map.shape)
            #................................................................construct save path.    changable
            filename = str(path).split('\\')[-1].split('.')[0] + '.mat'
            videoname = str(path).split('\\')[-3]
            labelname = str(path).split('\\')[-5]

            if not os.path.exists(os.path.join(dir,labelname,videoname)):
                os.makedirs(os.path.join(dir,labelname,videoname))

            output = os.path.join(dir, labelname,videoname,filename)

            scipy.io.savemat(output, mdict={'feature':feature_map})
            print("ok", output)




