import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from loader import get_loader
from solver import Solver

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--Loop_test', type=bool, default=False)#Test all the saved models
parser.add_argument('--saved_path', type=str, default='./npy_img/')
parser.add_argument('--save_path', type=str, default='./save/')
parser.add_argument('--result_fig', type=bool, default=True)

parser.add_argument('--norm_range_min', type=float, default=0)
parser.add_argument('--norm_range_max', type=float, default=255)
parser.add_argument('--trunc_min', type=float, default=0)
parser.add_argument('--trunc_max', type=float, default=255)


parser.add_argument('--patch_n', type=int, default=10)
parser.add_argument('--patch_size', type=int, default=55)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--num_epochs', type=int, default=505)
parser.add_argument('--print_iters', type=int, default=20)
parser.add_argument('--decay_epochs', type=int, default=20)
parser.add_argument('--save_epochs', type=int, default=20)
parser.add_argument('--test_epochs', type=int, default=500)

parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--device', type=str)
parser.add_argument('--multi_gpu', type=bool, default=True)


args = parser.parse_args(args=[])

def main(args):
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    print('Preparing data')
    data_loader = get_loader(mode=args.mode,#默认为train
                             saved_path=args.saved_path,#'./npy_img/'
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=args.patch_size,#
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             )
    print('The data is ready')
    solver = Solver(args, data_loader)
    print('Solver is ready')
    if args.mode == 'train':
        print('Lets start training.')
        solver.train()
    elif args.mode == 'test':
        print('Lets start testing.')
        solver.test()

main(args)
