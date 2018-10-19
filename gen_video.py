#!/usr/bin/env python3


'''
file aims to generate a txt, containing the image name and path.
catogory srotes labels, meaninglessly I think

input: path you desire to convert to txt
output: generate a txt file, which can be uesd as label,
and join the path, you can search images
'''






import os
import glob

path="G:\kk_file\EmotiW\AFEW_IMAGE_align_crop_feature\Data\\val"


catogory=[]

subfile = os.listdir(path)
for label in subfile:
    if os.path.isdir(path+'/'+label):    #G:/kk_file/EmotiW/AFEW_IMAGE/Train_image\Angry
        catogory.append(label)
        file = os.listdir(os.path.join(path,label))
        for num in file:
            print(os.path.join(label,num))         #Neutral\003322280
            #img = os.path.join(path,label,num)+'\\'+'*.png'
            #img = glob.glob(img)
            #for pic in img:

            content = os.path.join(label,num)

            file = open(path + '/video' + '.txt', 'a')
            file.write(content +'\n')
            file.close()


print("writing is done")
print(catogory)






















