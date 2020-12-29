import os
from datetime import datetime

import torch.multiprocessing as mp

from utils import Logger
from .ddp import dist_process
from .dp import process
from .test import test


class Trainer(object):
    def __init__(self, para):
        self.para = para
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '6666'

    def run(self):
        # recoding parameters
        self.para.time = datetime.now()
        logger = Logger(self.para)
        logger.record_para()

        # training
        if not self.para.test_only:
            if self.para.trainer_mode == 'ddp':
                gpus = self.para.num_gpus
                processes = []
                for rank in range(gpus):
                    p = mp.Process(target=dist_process, args=(rank, self.para))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            elif self.para.trainer_mode == 'dp':
                process(self.para)

        # test
        test(self.para, logger)
