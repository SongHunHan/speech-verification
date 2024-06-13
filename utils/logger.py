import wandb

class CustomLogger:
    def __init__(self, config):
        self.wandb_config = config['wandb']
        self.wandb_use = self.wandb_config['use']
        
        if self.wandb_use:
            self.wandb = wandb.init(project=self.wandb_config['project_name'])
            self.wandb.name = self.wandb_config['run_name']
            
    def watch(self, model):
        if self.wandb_use:
            self.wandb.watch(model)
            
    def log(self, log_dict):
        if self.wandb_use:
            self.wandb.log(log_dict)

    def logs(self, logs:dict, step:int):
        if self.use_wandb:
            for key, value in logs.items():
                self.log(key,value)
    
    
    def finish(self):
        if self.wandb_use:
            self.wandb.finish()
    
    