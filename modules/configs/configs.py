import json

class Configs():
    def __init__(self):
        
        with open('src/configs_json/config_train.json') as f:
            data = json.load(f)
        self.data = data
        self.batch_size = data['batch_size']
        self.num_workers = data['num_workers']
        self.num_epochs = data['num_epochs']
        self.patience = data['patience']
        self.lr = data['lr']
        self.device = data['device']
        self.seed = data['seed']
        self.backbone = data['backbone']
        self.num_classes = data['num_classes']
        self.model = data['model']
        self.path_info = data['path_info']
        if self.path_info:
            self.save_dir_model = f'{self.model}_{self.num_epochs}ep_{self.patience}p_{self.lr}lr_{self.path_info}'
        else:
            self.save_dir_model = f'{self.model}_{self.num_epochs}ep_{self.patience}p_{self.lr}lr'
        self.model_checkpoint = data['model_checkpoint']