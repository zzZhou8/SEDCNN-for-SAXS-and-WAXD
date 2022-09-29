import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from loader import get_loader
from solver import Solver

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--Loop_test', type=bool, default=False)#循环测试以绘制走势图
parser.add_argument('--saved_path', type=str, default='./npy_img/')
parser.add_argument('--save_path', type=str, default='./save/')
parser.add_argument('--result_fig', type=bool, default=True)

parser.add_argument('--norm_range_min', type=float, default=0)
parser.add_argument('--norm_range_max', type=float, default=255)
parser.add_argument('--trunc_min', type=float, default=0)
parser.add_argument('--trunc_max', type=float, default=255)


parser.add_argument('--patch_n', type=int, default=10)#一个图进行多少次随机切割，在随机切割时这个数最大为6，如果是大小切的话就是1没多的！
parser.add_argument('--patch_size', type=int, default=55)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--num_epochs', type=int, default=505)#决定训练轮次
parser.add_argument('--print_iters', type=int, default=20)
parser.add_argument('--decay_epochs', type=int, default=20)
parser.add_argument('--save_epochs', type=int, default=20)
parser.add_argument('--test_epochs', type=int, default=500)

parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--device', type=str)
parser.add_argument('--multi_gpu', type=bool, default=True)


args = parser.parse_args(args=[])

def main(args):
    
    if not os.path.exists(args.save_path):#没有就产生一个
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:#专门存结果的地方
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    print('正在准备数据啦')
    data_loader = get_loader(mode=args.mode,#默认为train
                             saved_path=args.saved_path,#'./npy_img/'
                             patch_n=(args.patch_n if args.mode=='train' else None),#
                             patch_size=args.patch_size,#
                             batch_size=(args.batch_size if args.mode=='train' else 1),#训练时为16，测试时为1
                             )
    print('数据准备好啦')
    solver = Solver(args, data_loader)
    print('solver准备好了')
    if args.mode == 'train':
        print('开始训练啦')
        solver.train()
    elif args.mode == 'test':
        print('开始测试啦')
        solver.test()

main(args)
