import os
import math
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
from networks import  SEDCNN_SAXS
from measure import compute_measure
from skimage import io

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from sklearn.metrics import r2_score

import pyFAI.azimuthalIntegrator as pyFAI

class Solver(object):
    def __init__(self,args,data_loader):
        self.mode=args.mode
        self.Loop_test=args.Loop_test

        self.data_loader=data_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#不说的话就默认

        #在这里的话，我们就要保持一个合适的缩放比了，因为所有的图我们都专门做了0-1的归一化，所有在这里我们可以把min给0，max给到255
        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path#存储结果
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_epochs = args.decay_epochs
        self.save_epochs = args.save_epochs
        self.test_epochs = args.test_epochs
        self.result_fig = args.result_fig

        self.SEDCNN_SAXS = SEDCNN_SAXS()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):#多gpu调用
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.SEDCNN_SAXS= nn.DataParallel(self.SEDCNN_SAXS)
        self.SEDCNN_SAXS.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.SEDCNN_SAXS.parameters(), self.lr)

    def save_model(self, epochs_):
        f = os.path.join(self.save_path, 'SEDCNN4_SAXS_{}epochs.ckpt'.format(epochs_))#按照轮次来存的
        torch.save(self.SEDCNN_SAXS.state_dict(), f)

    def load_model(self, epochs_):
        f = os.path.join(self.save_path, 'SEDCNN4_SAXS_{}epochs.ckpt'.format(epochs_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f).items():
                n = k[7:]#反正大家都这么写，凑合着往下看吧
                state_d[n] = v
            self.SEDCNN_SAXS.load_state_dict(state_d)
        else:
            self.SEDCNN_SAXS.load_state_dict(torch.load(f))

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

    def save_tif(self,pred,fig_name):

        fig_path = os.path.join(self.save_path, 'Revised_SAXS_tif')

        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

        pred=pred.numpy()

        io.imsave(os.path.join(fig_path, 'Revised_SAXS_{}.tif'.format(fig_name)),np.float32(pred))
    
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

    #开始炼丹了
    def train(self):
        train_losses=[]
        total_iters=0
        total_epochs=0
        start_time=time.time()
        for epoch in range(1, self.num_epochs):
            total_epochs += 1
            print(epoch)
            self.SEDCNN_SAXS.train(True)#理解成mode.train()即可
            
            for iter_, (x,y) in enumerate(self.data_loader):
                total_iters += 1

            # add 1 channel,转为灰度图要不然网络处理不了，但是给的颇显随意
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                size=x.size()[-1]#用于统一尺寸

                x = x.view(-1, 1, size, size)
                y = y.view(-1, 1, size, size)   
        
                pred = self.SEDCNN_SAXS(x)
                loss = self.criterion(pred, y) 
                self.SEDCNN_SAXS.zero_grad()#都是模型梯度归0的方式
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
            delat_mean_optimized_dot3_list=[]
            delta_intensity_optimized_dot3_list=[]
            delat_Var_optimized_dot3_list=[]
            dot3_direct_optimized_ava_list=[]

            loop_num=int(self.test_epochs/self.save_epochs)
            print('我们将对{}组模型进行测试'.format(loop_num))
            for idx in range(1,loop_num+1):
                
                print('正在对第{}组模型进行测试'.format(idx))

                #释放原本的模型
                del self.SEDCNN_SAXS

                #循环载入
                self.SEDCNN_SAXS = SEDCNN_SAXS().to(self.device)
                test_epoch=int(self.save_epochs*idx)
                self.load_model(test_epoch)#重新保存的模型

                # compute PSNR, SSIM, RMSE
                pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
                
                #计算指向差异的平均值
                dot3_direct_optimized_ava=0

                #计算优化后的方差，均值及强度的差异
                delat_mean_optimized_dot3=0
                delta_intensity_optimized_dot3=0
                delat_Var_optimized_dot3=0


                ai = pyFAI.AzimuthalIntegrator(wavelength=1.2461e-10)
                ai.setFit2D(directDist=3100, centerX=300 ,centerY=291,tilt=0, tiltPlanRotation=0, pixelX=75, pixelY=75, splineFile=None)
                caculate = ai.integrate_radial
                
                with torch.no_grad():
                    for i, (x,y) in enumerate(self.data_loader):
                        shape_ = x.shape[-1]#要图片的真实尺寸去了

                        x = x.unsqueeze(0).float().to(self.device)
                        y = y.unsqueeze(0).float().to(self.device)

                        pred = self.SEDCNN_SAXS(x)

                        # denormalize, truncate
                        x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                        y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                        pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                        I2,tth2=plot_dot(caculate,np.array(pred))
                        I3,tth3=plot_dot(caculate,np.array(y))

                        popt2,r2_2=return_popts(I2,tth2)
                        popt3,r2_3=return_popts(I3,tth3)

                        #计算取向角差异
                        delta_optimized_dot3=float(abs(direct(popt3)-direct(popt2)))

                        #对均值，强度，方差和取向角差值进行计算：
                        delat_mean_optimized_dot3 += delat_mean(popt2,popt3)
                        delta_intensity_optimized_dot3 += delat_intensity(popt2,popt3)
                        delat_Var_optimized_dot3 += delat_Var(popt2,popt3)
                        dot3_direct_optimized_ava += delta_optimized_dot3


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
                    delat_mean_optimized_dot3=delat_mean_optimized_dot3/len(self.data_loader)
                    delat_Var_optimized_dot3=delat_Var_optimized_dot3/len(self.data_loader)
                    delta_intensity_optimized_dot3=delta_intensity_optimized_dot3/len(self.data_loader)
                    dot3_direct_optimized_ava=dot3_direct_optimized_ava/len(self.data_loader)

                    #均值在列表中进行保存：
                    pred_psnr_avg_list.append(pred_psnr_avg)
                    pred_ssim_avg_list.append(pred_ssim_avg)
                    pred_rmse_avg_list.append(pred_rmse_avg)
                    delat_mean_optimized_dot3_list.append(delat_mean_optimized_dot3)
                    delat_Var_optimized_dot3_list.append(delat_Var_optimized_dot3)
                    delta_intensity_optimized_dot3_list.append(delta_intensity_optimized_dot3)
                    dot3_direct_optimized_ava_list.append(dot3_direct_optimized_ava)  
                    
                    print('\n')
                    #显示一下最后的训练效果呗
                    print('对于第{}保存的模型其预测结果为：'.format(test_epoch))
                    print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} '.format(pred_psnr_avg, pred_ssim_avg, pred_rmse_avg))
                    #展示一下各类差值

                    print('\n')
                    print('Optimized Dot3 === \nDelta Mean avg: {:.4f} \nDelta Var avg: {:.4f} \nDelta Intensity avg: {:.4f} \nDelta Direct avg: {:.4f}'.format(delat_mean_optimized_dot3, delat_Var_optimized_dot3, delta_intensity_optimized_dot3,dot3_direct_optimized_ava))
                
                #最终的数据保存
                avg_list='avg_list'
                if not os.path.exists(avg_list):#没有就产生一个
                    os.makedirs(avg_list)
                    print('Create path : {}'.format(avg_list))
                
                np.save(os.path.join(avg_list,'pred_psnr_avg_list.npy'),pred_psnr_avg_list)
                np.save(os.path.join(avg_list,'pred_ssim_avg_list.npy'),pred_ssim_avg_list)
                np.save(os.path.join(avg_list,'pred_rmse_avg_list.npy'),pred_rmse_avg_list)

                np.save(os.path.join(avg_list,'delat_mean_optimized_dot3_list.npy'),delat_mean_optimized_dot3_list)
                np.save(os.path.join(avg_list,'delat_Var_optimized_dot3_list.npy'),delat_Var_optimized_dot3_list)
                np.save(os.path.join(avg_list,'delta_intensity_optimized_dot3_list.npy'),delta_intensity_optimized_dot3_list)
                np.save(os.path.join(avg_list,'dot3_direct_optimized_ava_list.npy'),dot3_direct_optimized_ava_list)
        else:#直接测试
            del self.SEDCNN_SAXS#把原本的模型进行了释放
            #load
            self.SEDCNN_SAXS = SEDCNN_SAXS().to(self.device)
            self.load_model(self.test_epochs)#删除现在现有的模型，重新加载上一个要求轮次下保存的模型

            # compute PSNR, SSIM, RMSE
            ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
            pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
            
            #计算指向差异的平均值
            dot3_direct_ava=0
            dot3_optimized_ava=0

            #计算方差，均值及强度的差异
            #对均值差值进行计算：
            delat_mean_dot3=0
            delat_mean_optimized_dot3=0
            delat_mean_dot10=0
            
            #对强度差值进行计算：
            delta_intensity_dot3=0
            delta_intensity_optimized_dot3=0
            delta_intensity_dot10=0

            #对方差差值进行计算：
            delta_Var_dot3=0
            delat_Var_optimized_dot3=0
            delat_Var_dot10=0

            ai = pyFAI.AzimuthalIntegrator(wavelength=1.2461e-10)
            ai.setFit2D(directDist=3100, centerX=300 ,centerY=291,tilt=0, tiltPlanRotation=0, pixelX=75, pixelY=75, splineFile=None)
            caculate = ai.integrate_radial
            
            with torch.no_grad():
                for i, (x,y) in enumerate(self.data_loader):
                    shape_ = x.shape[-1]#要图片的真实尺寸去了

                    x = x.unsqueeze(1).float().to(self.device)
                    y = y.unsqueeze(1).float().to(self.device)

                    pred = self.SEDCNN_SAXS(x)

                    # denormalize, truncate
                    x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                    y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                    pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                    I1,tth1=plot_dot(caculate,np.array(x))
                    I2,tth2=plot_dot(caculate,np.array(pred))
                    I3,tth3=plot_dot(caculate,np.array(y))

                    popt1,r2_1=return_popts(I1,tth1)
                    popt2,r2_2=return_popts(I2,tth2)
                    popt3,r2_3=return_popts(I3,tth3)

                    delta_dot3,delta_optimized_dot3=plot_pics(tth1,I1,popt1,tth2,I2,popt2,tth3,I3,popt3,r2_1,r2_2,r2_3,i,self.save_path)

                    #对均值差值进行计算：
                    delat_mean_dot3+=delat_mean(popt1,popt3)
                    delat_mean_optimized_dot3+=delat_mean(popt2,popt3)
                    delat_mean_dot10+=delat_mean(popt3,popt3)
                    
                    #对强度差值进行计算：
                    delta_intensity_dot3+=delat_intensity(popt1,popt3)
                    delta_intensity_optimized_dot3+=delat_intensity(popt2,popt3)
                    delta_intensity_dot10+=delat_intensity(popt3,popt3)

                    #对方差差值进行计算：
                    delta_Var_dot3+=delat_Var(popt1,popt3)
                    delat_Var_optimized_dot3+=delat_Var(popt2,popt3)
                    delat_Var_dot10+=delat_Var(popt3,popt3)

                    dot3_direct_ava += delta_dot3
                    dot3_optimized_ava += delta_optimized_dot3

                    data_range = self.trunc_max - self.trunc_min

                    original_result, pred_result = compute_measure(x, y, pred, data_range)
                    ori_psnr_avg += original_result[0]
                    ori_ssim_avg += original_result[1]
                    ori_rmse_avg += original_result[2]
                    pred_psnr_avg += pred_result[0]
                    pred_ssim_avg += pred_result[1]
                    pred_rmse_avg += pred_result[2]

                    # save result figure
                    if self.result_fig:
                        self.save_fig(x, y, pred, i, original_result, pred_result)#属于作图给大家看一眼了
                    self.save_tif(pred,i)#如果是对原图做测试请用这个
                    
            
                    printProgressBar(i, len(self.data_loader),
                                    prefix="Compute measurements ..",
                                    suffix='Complete', length=25)#显示一下进度
                
                print('\n')
                #显示一下最后的训练效果呗
                print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}  \nDelta of diret ava: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                                ori_ssim_avg/len(self.data_loader), 
                                                                                                ori_rmse_avg/len(self.data_loader),
                                                                                                dot3_direct_ava/len(self.data_loader)))
                print('\n')
                print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} \nDelta of diret ava: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                    pred_ssim_avg/len(self.data_loader), 
                                                                                                    pred_rmse_avg/len(self.data_loader),
                                                                                                    dot3_optimized_ava/len(self.data_loader)))
                #展示一下各类差值
                print('\n')
                print('Dot3 === \nDelta Mean avg: {:.4f} \nDelta Var avg: {:.4f} \nDelta Intensity avg: {:.4f}'.format(delat_mean_dot3/len(self.data_loader), 
                                                                                                delta_Var_dot3/len(self.data_loader), 
                                                                                                delta_intensity_dot3/len(self.data_loader)))
                
                print('\n')
                print('Optimized Dot3 === \nDelta Mean avg: {:.4f} \nDelta Var avg: {:.4f} \nDelta Intensity avg: {:.4f}'.format(delat_mean_optimized_dot3/len(self.data_loader), 
                                                                                                delat_Var_optimized_dot3/len(self.data_loader), 
                                                                                                delta_intensity_optimized_dot3/len(self.data_loader)))

                print('\n')
                print('Sec1 === \nDelta Mean avg: {:.4f} \nDelta Var avg: {:.4f} \nDelta Intensity avg: {:.4f}'.format(delat_mean_dot10/len(self.data_loader), 
                                                                                                delat_Var_dot10/len(self.data_loader), 
                                                                                                delta_intensity_dot10/len(self.data_loader)))

def Determine_initial_parameters(I_one_cut,tth_all_cut):

        #####################从这里开始进行参数的拟合#####################
        ####对均值进行拟合，即预设峰值位置
        p2=[100,100,25,200,20,20]#(A,M,S)A_scale,M_mean,S_sigma
        #p2=[振幅1，振幅2，均值1，均值2，方差1，方差2]

        ###振幅设置为最大强度的90%
        p2[0]=max(I_one_cut)*0.9#第一个峰的强度值
        p2[1]=max(I_one_cut)*0.9#第二个峰的强度值

        ###对于tth角进行确定，均值就是那两个极大值的位置
        for tmp in range(10,100,10):###从10开始左右10个，20个，30个进行比较与寻峰。
            max_idxs=argrelextrema(I_one_cut, np.greater,order=tmp)###获取多个局部极值的索引值
            tths_of_max=tth_all_cut[max_idxs]#####找到索引相对应的tth们
            #####恰好能找到两个极值点的时候
            if len(tths_of_max)==2:
                break

        ###对于特殊的只有单个max的情况进行说明
        if len(tths_of_max)==1:
            one_max=tths_of_max[0]
            if one_max >= 0:
                tths_of_max=[one_max-180,one_max]
            else :
                tths_of_max=[one_max,one_max+180]

        ###什么也没有的那些图也给一个角度吧
        if len(tths_of_max)==0:
            tths_of_max=[-90,90]

        p2[2]=tths_of_max[0]
        p2[3]=tths_of_max[1]
    
        return p2
    
def gaussian2(x,*param):

    gauss_1=param[0]*np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.)))
    gauss_2=param[1]*np.exp(-np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))

    return gauss_1+gauss_2

def fit_curve(I_one_cut,tth_all_cut,p2):
    try: 
        popt,pcov = curve_fit(gaussian2,tth_all_cut,I_one_cut,p0=p2)##用这一套参数来拟合！
        ###########评价因子使用R2系数来评估拟合效果###########
        r2=r2_score(I_one_cut,gaussian2(tth_all_cut,*popt))
        delta_tth=abs(abs(popt[3]-popt[2])-180)
    except RuntimeError:
        ###拟合失败的就给一个这样的全0，和初始化的返回值
        r2=0####拟合度为0
        delta_tth=100#####角度差值大于100度
        popt=np.array(p2)

    return popt,r2,delta_tth

def return_popts(I1,tth1):
    p2_standing=Determine_initial_parameters(tth1,I1)#####确定初始化参数
    popt_standing,r2_standing,delta_tth_standing=fit_curve(I1,tth1,p2_standing)#####输出拟合参数，确定系数，tth的差值
    return popt_standing,r2_standing

def delat_intensity(popts2,popts3):
    return float(abs(popts2[0]-popts3[0])+abs(popts2[1]-popts3[1]))

def delat_mean(popts2,popts3):
    return float(abs(popts2[2]-popts3[2])+abs(popts2[3]-popts3[3]))
    
def delat_Var(popts2,popts3):
    return float(abs(popts2[4]-popts3[4])+abs(popts2[5]-popts3[5]))

def direct(popts):
    return float((popts[2]+popts[3]+180)/2)

def plot_dot(caculate,pic):
    
        rad_min=0.283
        rad_max=0.300
        delta_q=rad_max-rad_min

        detector_frames=pic

        tth, I_mid= caculate(detector_frames,npt_rad=1, npt=360,
        
        
                                                correctSolidAngle=True,unit="chi_deg",
                                                
                                                radial_unit ='q_nm^-1',
        
                                                azimuth_range=(-180.,180.),
        
                                                radial_range=(rad_min,rad_max)) 
        
        tth, I_inner= caculate(detector_frames,npt_rad=1, npt=360,
        
        
                                                correctSolidAngle=True,unit="chi_deg",
                                                
                                                radial_unit ='q_nm^-1',
        
                                                azimuth_range=(-180.,180.),
        
                                                radial_range=(rad_min-delta_q,rad_min)) 
        
        tth, I_outer= caculate(detector_frames,npt_rad=1, npt=360,
        
        
                                                correctSolidAngle=True,unit="chi_deg",
                                                
                                                radial_unit ='q_nm^-1',
        
                                                azimuth_range=(-180.,180.),
        
                                                radial_range=(rad_max,rad_max+delta_q))

        I=I_mid-(I_inner+I_outer)/2
        I[I<0]=0
        
        min_cut=np.where(tth<140)[-1][-1]#np.where()返回的是索引值所以，这就算是在偷巧了
        max_cut=np.where(tth>160)[0][0]

        I=np.delete(I,range(min_cut,max_cut))#删除靠近遮挡物部分的数据，这边必然会产生大量的失真    
        tth=np.delete(tth,range(min_cut,max_cut))#tth角这表也删除了
    
        unnormal_min_cut=np.where(tth<10)[-1][-1]
        unnormal_max_cut=np.where(tth>50)[0][0]

        I=np.delete(I,range(unnormal_min_cut,unnormal_max_cut))
        tth=np.delete(tth,range(unnormal_min_cut,unnormal_max_cut))

        unnormal_min_cut_2=np.where(tth<-50)[-1][-1]
        unnormal_max_cut_2=np.where(tth>-10)[0][0]

        I=np.delete(I,range(unnormal_min_cut_2,unnormal_max_cut_2))
        tth=np.delete(tth,range(unnormal_min_cut_2,unnormal_max_cut_2))

        return I,tth

def plot_pics(tth_1,I_1,popts1,tth_2,I_2,popts2,tth_3,I_3,popts3,r2_1,r2_2,r2_3,idx,save_path):

        f, ax = plt.subplots(1, 3, figsize=(40, 10))#画1*3个子图出来，每一个子图的尺寸为40*20
        path3 = os.path.join(save_path, 'Dots_curve')

        x=np.array(list(range(math.floor(min(tth_1)),math.ceil(max(tth_1)))))

        #指向角计算
        Diret1=(popts1[2]+popts1[3]+180)/2#原始图的
        Diret2=(popts2[2]+popts2[3]+180)/2#预测图的
        Diret3=(popts3[2]+popts3[3]+180)/2#自己的

        #计算原图与1s的指向角差异
        delta_of_dot3_and_sec1=abs(Diret3-Diret1)
        delta_of_optimized_dot3_and_sec1=abs(Diret3-Diret2)

        #对均值差值进行计算：
        delat_mean_dot3=delat_mean(popts1,popts3)
        delat_mean_optimized_dot3=delat_mean(popts2,popts3)
        delat_mean_dot10=delat_mean(popts3,popts3)
        
        #对强度差值进行计算：
        delta_intensity_dot3=delat_intensity(popts1,popts3)
        delta_intensity_optimized_dot3=delat_intensity(popts2,popts3)
        delta_intensity_dot10=delat_intensity(popts3,popts3)

        #对方差差值进行计算：
        delta_Var_dot3=delat_Var(popts1,popts3)
        delta_Var_optimized_dot3=delat_Var(popts2,popts3)
        delta_Var_dot10=delat_Var(popts3,popts3)

        #散点图
        ax[0].plot(tth_1,I_1,'*')
        ax[0].plot(x,gaussian2(x,*popts1),'r.')
        ax[0].set_title('Raw_0.3s_SAXS', fontsize=30)#抬头的标题
        ax[0].set_xlabel("Diret_angle: {:.4f}{}Delta_angle: {:.4f}\nDelta_mean: {:.4f}{}Delta_Var: {:.4f}{}Delta_Intensity: {:.4f}".format(Diret1,' ',delta_of_dot3_and_sec1,
                                                                                                delat_mean_dot3,' ',
                                                                                                delta_Var_dot3,' ',
                                                                                                delta_intensity_dot3,), fontsize=20)

        ax[1].plot(tth_2,I_2,'*')
        ax[1].plot(x,gaussian2(x,*popts2),'r.')
        ax[1].set_title('Optimized_0.3s_SAXS', fontsize=30)#抬头的标题
        ax[1].set_xlabel("Diret_angle: {:.4f}{}Delta_angle: {:.4f}\nDelta_mean: {:.4f}{}Delta_Var: {:.4f}{}Delta_Intensity: {:.4f}".format(Diret2,' ',delta_of_optimized_dot3_and_sec1,
                                                                                                delat_mean_optimized_dot3,' ',
                                                                                                delta_Var_optimized_dot3,' ',
                                                                                                delta_intensity_optimized_dot3,), fontsize=20)

        ax[2].plot(tth_3,I_3,'*')
        ax[2].plot(x,gaussian2(x,*popts3),'r.')
        ax[2].set_title('Raw_1s_SAXS', fontsize=30)#抬头的标题
        ax[2].set_xlabel("Diret_angle: {:.4f}{}Delta_angle: {:.4f}\nDelta_mean: {:.4f}{}Delta_Var: {:.4f}{}Delta_Intensity: {:.4f}".format(Diret3,' ',0,
                                                                                                delat_mean_dot10,' ',
                                                                                                delta_Var_dot10,' ',
                                                                                                delta_intensity_dot10,), fontsize=20)

        if not os.path.exists(path3):#没有就产生一个
            os.makedirs(path3)
            print('Create path : {}'.format(path3))

        f.savefig(os.path.join(path3, 'curve_of_{}.png'.format(idx)))

        plt.close()

        return delta_of_dot3_and_sec1,delta_of_optimized_dot3_and_sec1