import os
from glob import glob
import numpy as np
from torch.utils.data import DataLoader,Dataset
import random

#这里的测试方式还是比较属于自导自演的，因为是自己测试自己的未知，用以评估系统自身的降噪能力
class SAXS_dataset(Dataset):
    def __init__(self,mode,saved_path,patch_n=None,patch_size=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"#确定现在是在测试还是在训练
        
        train_input_path=os.path.join(saved_path,'SAXS_train_dot3.npy')
        train_target_path=os.path.join(saved_path,'SAXS_train_dot10.npy')

        test_input_path=os.path.join(saved_path,'SAXS_test_dot3.npy')
        test_target_path=os.path.join(saved_path,'SAXS_test_dot10.npy')

        self.mode=mode
        self.patch_n = patch_n
        self.patch_size = patch_size

        if self.mode=='train':
            #这里是把最后的两组图片完全拿来测试了，用以测试模型的降噪能力是否合乎规范
            self.input_=np.load(train_input_path)##这就是纯粹因为网络架构的原因妥协了576=32*18,所以遭得住这么折腾
            self.target_=np.load(train_target_path)

        else: # self.mode =='test'
            
            self.input_=np.load(test_input_path)
            self.target_=np.load(test_target_path)
    


    def __len__(self):
        return self.input_.shape[0]

    def __getitem__(self, idx):

        if self.mode=='train':

            input_img=self.input_[idx]
            target_img=self.target_[idx]

            if self.patch_size:
                input_patch,target_patch=get_patch(input_img,target_img,self.patch_n,self.patch_size)
                return (input_patch,target_patch)#训练时是返回切割的数据的
            else:
                return(input_img,target_img)#test的时候是原封不动的返回的

        else: # mode =='test'

            input_img=self.input_[idx]
            target_img=self.target_[idx]

            return(input_img,target_img)#test的时候是原封不动的返回的
def get_patch(full_input_img,full_target_img,patch_n,patch_size):#10,64
    assert full_input_img.shape == full_target_img.shape#尺寸要对的上才行
    patch_input_imgs=[]
    patch_target_imgs=[]
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size#决定了裁切后的图片有多大
    for _ in range(patch_n):#决定要产生多少幅64*64的训练用图片
        top=np.random.randint(0,h-new_h)#在这两个闭区间中取整数
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]#抽取的很随机嘛，但是数量有限
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)

def get_loader(mode='train',saved_path=None,patch_n=None,patch_size=None,batch_size=32):
    dataset_=SAXS_dataset(mode,saved_path,patch_n,patch_size)
    dataloader=DataLoader(dataset=dataset_,batch_size=batch_size,shuffle=(True if mode=='train' else False))
    return dataloader


