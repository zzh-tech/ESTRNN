import argparse


class Parameter:
    def __init__(self):
        self.args = self.extract_args()

    def extract_args(self):
        self.parser = argparse.ArgumentParser(description='Video Deblurring')

        # experiment mark
        self.parser.add_argument('--description', type=str, default='develop', help='experiment description')

        # global parameters
        self.parser.add_argument('--seed', type=int, default=39, help='random seed')
        self.parser.add_argument('--threads', type=int, default=8, help='# of threads for dataloader')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='# of GPUs to use')
        self.parser.add_argument('--no_profile', action='store_true', help='show # of parameters and computation cost')
        self.parser.add_argument('--profile_H', type=int, default=720,
                                 help='height of image to generate profile of model')
        self.parser.add_argument('--profile_W', type=int, default=1280,
                                 help='width of image to generate profile of model')
        self.parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_file', type=str, default='', help='the path of checkpoint file for resume')

        # data parameters
        self.parser.add_argument('--data_root', type=str, default='/home/zhong/Dataset/', help='the path of dataset')
        self.parser.add_argument('--dataset', type=str, default='BSD', help='BSD, gopro_ds_lmdb, reds_lmdb')
        self.parser.add_argument('--save_dir', type=str, default='./experiment/',
                                 help='directory to save logs of experiments')
        self.parser.add_argument('--frames', type=int, default=8, help='# of frames of subsequence')
        self.parser.add_argument('--ds_config', type=str, default='2ms16ms', help='1ms8ms, 2ms16ms or 3ms24ms')
        self.parser.add_argument('--data_format', type=str, default='RGB', help='RGB or RAW')
        self.parser.add_argument('--patch_size', type=int, nargs='*', default=[256, 256])

        # model parameters
        self.parser.add_argument('--model', type=str, default='ESTRNN', help='type of model to construct')
        self.parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
        self.parser.add_argument('--n_blocks', type=int, default=15, help='# of blocks in middle part of the model')
        self.parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
        self.parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
        self.parser.add_argument('--activation', type=str, default='gelu', help='activation function')

        # loss parameters
        self.parser.add_argument('--loss', type=str, default='1*L1_Charbonnier_loss_color',
                                 help='type of loss function, e.g. 1*MSE|1e-4*Perceptual')

        # metrics parameters
        self.parser.add_argument('--metrics', type=str, default='PSNR', help='type of evaluation metrics')

        # optimizer parameters
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='method of optimization')
        self.parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        self.parser.add_argument('--lr_scheduler', type=str, default='cosine',
                                 help='learning rate adjustment stratedy')
        self.parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        self.parser.add_argument('--milestones', type=int, nargs='*', default=[200, 400])
        self.parser.add_argument('--decay_gamma', type=float, default=0.5, help='decay rate')

        # training parameters
        self.parser.add_argument('--start_epoch', type=int, default=1, help='first epoch number')
        self.parser.add_argument('--end_epoch', type=int, default=500, help='last epoch number')
        self.parser.add_argument('--trainer_mode', type=str, default='dp',
                                 help='trainer mode: distributed data parallel (ddp) or data parallel (dp)')

        # test parameters
        self.parser.add_argument('--test_only', action='store_true', help='only do test')
        self.parser.add_argument('--test_frames', type=int, default=20,
                                 help='frame size for test, if GPU memory is small, please reduce this value')
        self.parser.add_argument('--test_save_dir', type=str, help='where to save test results')
        self.parser.add_argument('--test_checkpoint', type=str,
                                 default='./model/checkpoints/model_best.pth.tar',
                                 help='the path of checkpoint file for test')
        self.parser.add_argument('--video', action='store_true', help='if true, generate video results')

        args, _ = self.parser.parse_known_args()

        args.normalize = True
        args.centralize = True

        return args
