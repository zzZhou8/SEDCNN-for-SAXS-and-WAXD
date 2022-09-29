import os
import numpy as np
from torch.utils.data import DataLoader,Dataset

#这里的测试方式还是比较属于自导自演的，因为是自己测试自己的未知，用以评估系统自身的降噪能力
class SAXS_dataset(Dataset):
    def __init__(self,mode,saved_path,Loop_test=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"#确定现在是在测试还是在训练
        
        train_input_path=os.path.join(saved_path,'WAXD_train_dot3.npy')
        train_target_path=os.path.join(saved_path,'WAXD_train_dot10.npy')

        test_input_path=os.path.join(saved_path,'WAXD_test_dot3.npy')
        test_target_path=os.path.join(saved_path,'WAXD_test_dot10.npy')

        self.mode=mode
        self.Loop_test=Loop_test

        if self.mode=='train':
            #这里是把最后的两组图片完全拿来测试了，用以测试模型的降噪能力是否合乎规范
            self.input_=np.load(train_input_path)
            self.target_=np.load(train_target_path)

        else: # self.mode =='test'
            self.input_=np.load(test_input_path)
            self.target_=np.load(test_target_path)


    def __len__(self):
        return self.input_.shape[0]

    def __getitem__(self, idx):

        if self.mode=='train':
            input_img=self.input_[idx]#这个还是一幅幅的在处理
            target_img=self.target_[idx]

            return (input_img,target_img)#训练时是返回切割的数据的

        else: # self.mode =='test'

            input_img=self.input_[idx]
            target_img=self.target_[idx]

            return(input_img,target_img)#test的时候是原封不动的返回的

        
def get_loader(mode='train',saved_path=None,Loop_test=None,batch_size=32):
    dataset_=SAXS_dataset(mode,saved_path,Loop_test)
    dataloader=DataLoader(dataset=dataset_,batch_size=batch_size,shuffle=(True if mode=='train' else False))
    return dataloader


