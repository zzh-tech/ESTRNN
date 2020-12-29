from importlib import import_module


class Data:
    def __init__(self, para, device_id):
        dataset = para.dataset
        module = import_module('data.' + dataset)
        self.dataloader_train = module.Dataloader(para, device_id, ds_type='train')
        self.dataloader_valid = module.Dataloader(para, device_id, ds_type='valid')
