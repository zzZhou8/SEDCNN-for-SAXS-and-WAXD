import xlwt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

f = xlwt.Workbook('encoding = utf-8')
sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) #创建sheet工作表

#横轴为训练轮次
write_num=25#一共保存多少轮
save_epochs=20#单词保存相隔的Epochs
epochs = np.linspace(save_epochs, save_epochs*write_num, write_num)

#统计结果
PSNR=np.load('pred_psnr_avg_list.npy').tolist()[:write_num]
SSIM=np.load('pred_ssim_avg_list.npy').tolist()[:write_num]
RMSE=np.load('pred_rmse_avg_list.npy').tolist()[:write_num]
Delta_direct_angle=np.load('dot3_direct_optimized_ava_list.npy').tolist()[:write_num]
Delta_mean=np.load('delat_mean_optimized_dot3_list.npy').tolist()[:write_num]
Delta_var=np.load('delat_Var_optimized_dot3_list.npy').tolist()[:write_num]
Delta_intensity=np.load('delta_intensity_optimized_dot3_list.npy').tolist()[:write_num]

mean_dot3,mean_1=2.1495,0
var_dot3,var_1=4.0267,0
inten_dot3,inten_1=3.5444,0
direct_dot3,direct_1=0.8668,0


def betas_caculate(Delta_direct_angle,direct_dot3,direct_1,weight):
    betas=[]
    denominator=direct_dot3-direct_1
    for data in Delta_direct_angle:
        Numerator=direct_dot3-data
        beta=(Numerator/denominator)*weight
        betas.append(beta)
    return betas

betas_direct_angle=betas_caculate(Delta_direct_angle,direct_dot3,direct_1,0.5)
betas_mean=betas_caculate(Delta_mean,mean_dot3,mean_1,0.3)
betas_var=betas_caculate(Delta_var,var_dot3,var_1,0.1)
betas_intensity=betas_caculate(Delta_intensity,inten_dot3,inten_1,0.1)

betas=[]
for idx in range (len(betas_direct_angle)):
    beta=betas_direct_angle[idx]+betas_mean[idx]+betas_var[idx]+betas_intensity[idx]
    betas.append(beta)
#print(betas)

all_data=[epochs,PSNR,SSIM,RMSE,Delta_direct_angle,Delta_mean,Delta_var,Delta_intensity,betas]
all_title=['epochs','PSNR','SSIM','RMSE','Delta_direct_angle','Delta_mean','Delta_var','Delta_intensity','betas']

for lie in range(len(all_data)):
    sheet1.write(0,lie,all_title[lie])
    for hang in range(len(all_data[1])): 
        sheet1.write(hang+1,lie,all_data[lie][hang]) #写入数据参数对应 行, 列, 值
f.save('All_Data.xls')#保存.xls到当前工作目录


all_Delta=[PSNR,SSIM,RMSE,Delta_direct_angle,Delta_mean,Delta_var,Delta_intensity]
all_Delta=np.array(all_Delta)
pMatric=np.corrcoef(all_Delta)#相关系数计算
pMatric=np.around(pMatric,decimals=3)#保留三位小数
fig, ax = plt.subplots(figsize = (9,9))
sns.heatmap(pMatric,annot=True, vmax=1, square=True, cmap="Reds")
ax.set_title('Parametric heat map', fontsize = 25)
ax.set_yticklabels(['PSNR','SSIM','RMSE','Delta direct angle','Delta mean','Delta var','Delta intensity'], fontsize = 12, rotation = 360, horizontalalignment='right')
ax.set_xticklabels(['PSNR','SSIM','RMSE','Delta direct angle','Delta mean','Delta var','Delta intensity'], fontsize = 12, horizontalalignment='right')
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.savefig("Heat_Map.png", dpi=300, quality=95,bbox_inches = 'tight')
plt.show()
