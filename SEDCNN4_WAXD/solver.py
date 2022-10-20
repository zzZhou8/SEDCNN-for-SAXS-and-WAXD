import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_epochs = args.decay_epochs
        self.save_epochs = args.save_epochs
        self.test_epochs= args.test_epochs
        self.result_fig = args.result_fig

        self.Loop_test = args.Loop_test

        self.SEDCNN_WAXD = SEDCNN_WAXD()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.SEDCNN_WAXD = nn.DataParallel(self.SEDCNN_WAXD)
        self.SEDCNN_WAXD.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.SEDCNN_WAXD.parameters(), self.lr)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'SEDCNN4_WAXD_{}epochs.ckpt'.format(iter_))
        torch.save(self.SEDCNN_WAXD.state_dict(), f)

    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'SEDCNN4_WAXD_{}epochs.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f).items():
                n = k[7:]
                state_d[n] = v
            self.SEDCNN_WAXD.load_state_dict(state_d)
        else:
            self.SEDCNN_WAXD.load_state_dict(torch.load(f))

    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self,x,y,pred,fig_name,original_result,pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))

        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Noise_pics', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],original_result[1],original_result[2]), fontsize=20)

        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Predict_pics', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        
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
    
    
    def train(self):
        train_losses=[]
        total_iters=0
        total_epochs=0
        start_time=time.time()
        for epoch in range(1, self.num_epochs):
            total_epochs += 1
            print(epoch)
            self.SEDCNN_WAXD.train(True)
            
            for iter_, (x,y) in enumerate(self.data_loader):
                total_iters += 1

            # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred = self.SEDCNN_WAXD(x)
                loss = self.criterion(pred, y) 
                self.SEDCNN_WAXD.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

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
 


    def test(self):
        if self.Loop_test:
            pred_psnr_avg_list=[]
            pred_ssim_avg_list=[]
            pred_rmse_avg_list=[]
            loop_num=int(self.test_epochs/self.save_epochs)
            print('We will test {} set of models'.format(loop_num))
            for idx in range(1,loop_num+1):
                print('We are testing the {} set of models'.format(idx))
                
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
                        shape_2 = x.shape[-1]

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
                                        suffix='Complete', length=25)

                    pred_psnr_avg=pred_psnr_avg/len(self.data_loader)
                    pred_ssim_avg=pred_ssim_avg/len(self.data_loader)
                    pred_rmse_avg=pred_rmse_avg/len(self.data_loader)

                    pred_psnr_avg_list.append(pred_psnr_avg)
                    pred_ssim_avg_list.append(pred_ssim_avg)
                    pred_rmse_avg_list.append(pred_rmse_avg)

                    print('\n')
                    print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg, pred_ssim_avg, pred_rmse_avg))

  
                avg_list='avg_list'
                if not os.path.exists(avg_list):
                    os.makedirs(avg_list)
                    print('Create path : {}'.format(avg_list))
                
                np.save(os.path.join(avg_list,'pred_psnr_avg_list.npy'),pred_psnr_avg_list)
                np.save(os.path.join(avg_list,'pred_ssim_avg_list.npy'),pred_ssim_avg_list)
                np.save(os.path.join(avg_list,'pred_rmse_avg_list.npy'),pred_rmse_avg_list)

        else:
            del self.SEDCNN_WAXD
            #load
            self.SEDCNN_WAXD = SEDCNN_WAXD().to(self.device)
            self.load_model(self.test_epochs)

            # compute PSNR, SSIM, RMSE
            ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
            pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
            
            with torch.no_grad():
                for i, (x,y) in enumerate(self.data_loader):
                    shape_1 = x.shape[-2]
                    shape_2 = x.shape[-1]

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

                    self.save_fig(x, y, pred, i, original_result, pred_result)
                    self.save_pre(pred,i)
            
                    printProgressBar(i, len(self.data_loader),
                                    prefix="Compute measurements ..",
                                    suffix='Complete', length=25)
                
                print('\n')
                
                print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                                ori_ssim_avg/len(self.data_loader), 
                                                                                                ori_rmse_avg/len(self.data_loader)))
                print('\n')
                print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                    pred_ssim_avg/len(self.data_loader), 
                                                                                                    pred_rmse_avg/len(self.data_loader)))
