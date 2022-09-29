import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from loader import get_loader
from solver import Solver

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--Loop_test', type=bool, default=True)#决定是否循环测试
parser.add_argument('--result_fig', type=bool, default=True)
parser.add_argument('--save_epochs', type=int, default=20)#460
parser.add_argument('--test_epochs', type=int, default=500)#23000
parser.add_argument('--decay_epochs', type=int, default=20)
parser.add_argument('--multi_gpu', type=bool, default=True)

parser.add_argument('--norm_range_min', type=float, default=0)
parser.add_argument('--norm_range_max', type=float, default=255)
parser.add_argument('--trunc_min', type=float, default=0)
parser.add_argument('--trunc_max', type=float, default=255)

parser.add_argument('--transform', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--saved_path', type=str, default='./npy_img/')
parser.add_argument('--save_path', type=str, default='./save/')
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--print_iters', type=int, default=20)


parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str)

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
                             Loop_test=(args.Loop_test if args.mode=='test' else None),#训练时为否，测试为可选激活项
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

