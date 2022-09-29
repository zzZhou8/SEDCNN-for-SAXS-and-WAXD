import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')#相当于暂且先别显示了
import matplotlib.pyplot as plt
#相当于让字典的存储是按照输入的顺序来存储的，而非最开始的乱序，现阶段好像已经没有这个问题了(一个已经是过去式的问题了)
from collections import OrderedDict
from readfiles import See_loss

import torch
import torch.nn as nn
import torch.optim as optim

from readfiles import printProgressBar
from networks import SEDCNN_WAXD
from measure import compute_measure
from skimage import io


class Solver(object):
    def __init__(self,args,data_loader):
        self.mode=args.mode
        self.data_loader=data_loader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#不说的话就默认

        #在这里的话，我们就要保持一个合适的缩放比了，因为所有的图我们都专门做了0-1的归一化，所有在这里我们可以把min给0，max给到60000
        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path#新路径下存储结果
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_epochs = args.decay_epochs
        self.save_epochs = args.save_epochs
        self.test_epochs= args.test_epochs
        self.result_fig = args.result_fig

        self.Loop_test = args.Loop_test

        self.SEDCNN_WAXD = SEDCNN_WAXD()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):#多gpu调用
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.SEDCNN_WAXD = nn.DataParallel(self.SEDCNN_WAXD)
        self.SEDCNN_WAXD.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.SEDCNN_WAXD.parameters(), self.lr)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'SEDCNN4_WAXD_{}epochs.ckpt'.format(iter_))#按照轮次来存的
        torch.save(self.SEDCNN_WAXD.state_dict(), f)

    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'SEDCNN4_WAXD_{}epochs.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f).items():
                n = k[7:]#反正大家都这么写，凑合着往下看吧
                state_d[n] = v
            self.SEDCNN_WAXD.load_state_dict(state_d)
        else:
            self.SEDCNN_WAXD.load_state_dict(torch.load(f))

    def lr_decay(self):
        lr = self.lr * 0.5#损失一半的学习率，先快后慢
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def denormalize_(self, image):###这个函数相当于在放大图像
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):#给个上下限呗
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self,x,y,pred,fig_name,original_result,pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))#画1*3个子图出来，每一个子图的尺寸为30*10

        #训练图片的
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Noise_pics', fontsize=30)#抬头的标题
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],original_result[1],original_result[2]), fontsize=20)#标一下图片的各项参数

        #预测图片的
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Predict_pics', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        #真实图片的
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('True_pics', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()

    def save_pre(self,pred,fig_name):

        fig_path = os.path.join(self.save_path, 'Revised_WAXD_tif')

        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
            
        pred=pred.numpy()
        io.imsave(os.path.join(fig_path , 'Revised_WAXD_{}.tif'.format(fig_name)),np.float32(pred))
    
    
    #开始炼丹了
    def train(self):
        train_losses=[]
        total_iters=0
        total_epochs=0
        start_time=time.time()
        for epoch in range(1, self.num_epochs):
            total_epochs += 1
            print(epoch)
            self.SEDCNN_WAXD.train(True)#理解成mode.train()即可
            
            for iter_, (x,y) in enumerate(self.data_loader):
                total_iters += 1

            # add 1 channel,转为灰度图要不然网络处理不了，但是给的颇显随意
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred = self.SEDCNN_WAXD(x)
                loss = self.criterion(pred, y) 
                self.SEDCNN_WAXD.zero_grad()#都是模型梯度归0的方式
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())#记录loss

                # print
                if total_iters % self.print_iters == 0:#到达输出轮数的倍数了
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))

                 # learning rate decay
                if total_iters % (self.decay_epochs*len(self.data_loader)) == 0:
                    self.lr_decay()

                # save model
                if total_iters % (self.save_epochs*len(self.data_loader)) == 0:
                    self.save_model(total_epochs)
                    np.save(os.path.join(self.save_path, 'loss_{}_epochs.npy'.format(total_epochs)), np.array(train_losses))
                    #相当于存到这个专属路径下的这个文件中去

    #所以很离谱的是训练的时候用的是裁切后的图片，而测试时用的是全尺寸的图片emmm
    def test(self):
        if self.Loop_test:
            pred_psnr_avg_list=[]
            pred_ssim_avg_list=[]
            pred_rmse_avg_list=[]
            loop_num=int(self.test_epochs/self.save_epochs)
            print('我们将对{}组模型进行测试'.format(loop_num))
            for idx in range(1,loop_num+1):
                print('正在对第{}组模型进行测试'.format(idx))
                
                #把原本的模型进行了释放
                del self.SEDCNN_WAXD

                #循环载入
                self.SEDCNN_WAXD = SEDCNN_WAXD().to(self.device)
                self.SEDCNN_WAXD.eval()
                test_epochs=int(self.save_epochs*idx)
                self.load_model(test_epochs)#重新保存的模型
                # compute PSNR, SSIM, RMSE
                pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

                with torch.no_grad():
                    for i, (x, y) in enumerate(self.data_loader):
                        shape_1 = x.shape[-2]
                        shape_2 = x.shape[-1]#要图片的真实尺寸去了

                        x = x.unsqueeze(1).float().to(self.device)
                        y = y.unsqueeze(1).float().to(self.device)

                        pred = self.SEDCNN_WAXD(x)

                        # denormalize, truncate
                        x = self.trunc(self.denormalize_(x.view(shape_1, shape_2).cpu().detach()))
                        y = self.trunc(self.denormalize_(y.view(shape_1, shape_2).cpu().detach()))
                        pred = self.trunc(self.denormalize_(pred.view(shape_1, shape_2).cpu().detach()))

                        data_range = self.trunc_max - self.trunc_min

                        original_result, pred_result = compute_measure(x, y, pred, data_range)

                        pred_psnr_avg += pred_result[0]
                        pred_ssim_avg += pred_result[1]
                        pred_rmse_avg += pred_result[2]

                        printProgressBar(i, len(self.data_loader),
                                        prefix="Compute measurements ..",
                                        suffix='Complete', length=25)#显示一下进度

                    #均值计算：
                    pred_psnr_avg=pred_psnr_avg/len(self.data_loader)
                    pred_ssim_avg=pred_ssim_avg/len(self.data_loader)
                    pred_rmse_avg=pred_rmse_avg/len(self.data_loader)

                    #均值在列表中进行保存：
                    pred_psnr_avg_list.append(pred_psnr_avg)
                    pred_ssim_avg_list.append(pred_ssim_avg)
                    pred_rmse_avg_list.append(pred_rmse_avg)

                    print('\n')
                    print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg, pred_ssim_avg, pred_rmse_avg))

                #最终的数据保存每轮次均保存
                avg_list='avg_list'
                if not os.path.exists(avg_list):#没有就产生一个
                    os.makedirs(avg_list)
                    print('Create path : {}'.format(avg_list))
                
                np.save(os.path.join(avg_list,'pred_psnr_avg_list.npy'),pred_psnr_avg_list)
                np.save(os.path.join(avg_list,'pred_ssim_avg_list.npy'),pred_ssim_avg_list)
                np.save(os.path.join(avg_list,'pred_rmse_avg_list.npy'),pred_rmse_avg_list)

        else:
            del self.SEDCNN_WAXD#把原本的模型进行了释放
            #load
            self.SEDCNN_WAXD = SEDCNN_WAXD().to(self.device)
            self.load_model(self.test_epochs)#删除现在现有的模型，重新加载上一个要求轮次下保存的模型

            # compute PSNR, SSIM, RMSE
            ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
            pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
            
            with torch.no_grad():
                for i, (x,y) in enumerate(self.data_loader):
                    shape_1 = x.shape[-2]
                    shape_2 = x.shape[-1]#要图片的真实尺寸去了

                    x = x.unsqueeze(1).float().to(self.device)
                    y = y.unsqueeze(1).float().to(self.device)

                    pred = self.SEDCNN_WAXD(x)

                    # denormalize, truncate
                    x = self.trunc(self.denormalize_(x.view(shape_1, shape_2).cpu().detach()))
                    y = self.trunc(self.denormalize_(y.view(shape_1, shape_2).cpu().detach()))
                    pred = self.trunc(self.denormalize_(pred.view(shape_1, shape_2).cpu().detach()))

                    data_range = self.trunc_max - self.trunc_min

                    original_result, pred_result = compute_measure(x, y, pred, data_range)
                    ori_psnr_avg += original_result[0]
                    ori_ssim_avg += original_result[1]
                    ori_rmse_avg += original_result[2]
                    pred_psnr_avg += pred_result[0]
                    pred_ssim_avg += pred_result[1]
                    pred_rmse_avg += pred_result[2]

                    self.save_fig(x, y, pred, i, original_result, pred_result)#属于作图给大家看一眼
                    self.save_pre(pred,i)#如果是对要测试图测试请用这个
            
                    printProgressBar(i, len(self.data_loader),
                                    prefix="Compute measurements ..",
                                    suffix='Complete', length=25)#显示一下进度
                
                print('\n')
                #显示一下最后的训练效果呗
                print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                                ori_ssim_avg/len(self.data_loader), 
                                                                                                ori_rmse_avg/len(self.data_loader)))
                print('\n')
                print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                    pred_ssim_avg/len(self.data_loader), 
                                                                                                    pred_rmse_avg/len(self.data_loader)))