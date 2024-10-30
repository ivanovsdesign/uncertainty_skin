from pytorch_lightning.loggers import Logger
from clearml import Task

class ClearMLLogger(Logger):
    def __init__(self, project_name, task_name, offline: bool = False):
        super().__init__()

        if offline == True: 
            self.set_offline()
        self._task = Task.init(project_name=project_name, 
                               task_name=task_name,
                               auto_connect_frameworks={
                                   'pytorch':['*.pt']
                               })

    @property
    def experiment(self):
        return self._task

    @property
    def version(self):
        return self._task.id
    
    def set_offline(self):
        Task.set_offline(True)

    def log_hyperparams(self, params):
        self._task.connect(params)

    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            self._task.get_logger().report_scalar(title='Metrics', series=key, value=value, iteration=step)
    
    def name(self):
        return "ClearMLLogger"