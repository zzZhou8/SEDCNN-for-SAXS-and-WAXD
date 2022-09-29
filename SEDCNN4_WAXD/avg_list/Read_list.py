import xlwt
import numpy as np
import matplotlib.pyplot as plt


f = xlwt.Workbook('encoding = utf-8')
sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) #创建sheet工作表

write_num=25#一共保存多少轮
save_epochs=20#单词保存相隔的Epochs
Epochs = np.linspace(save_epochs, save_epochs*write_num, write_num)
#横轴为训练轮次

#统计结果
PSNR=np.load('pred_psnr_avg_list.npy').tolist()[:write_num]
SSIM=np.load('pred_ssim_avg_list.npy').tolist()[:write_num]
RMSE=np.load('pred_rmse_avg_list.npy').tolist()[:write_num]


all_data=[Epochs,PSNR,SSIM,RMSE]
all_title=['Epochs','PSNR','SSIM','RMSE']

for lie in range(len(all_data)):
    sheet1.write(0,lie,all_title[lie])
    for hang in range(len(all_data[0])): 
        sheet1.write(hang+1,lie,all_data[lie][hang]) #写入数据参数对应 行, 列, 值
f.save('All_data.xls')#保存.xls到当前工作目录



