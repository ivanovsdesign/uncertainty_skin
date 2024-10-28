from pytorch_lightning.loggers import Logger
from clearml import Task

class ClearMLLogger(Logger):
    def __init__(self, project_name, task_name):
        super().__init__()
        self._task = Task.init(project_name=project_name, 
                               task_name=task_name,
                               auto_connect_frameworks={
                                   'pytorch':['*.ckpt', '*.pth']
                               })

    @property
    def experiment(self):
        return self._task

    @property
    def version(self):
        return self._task.id

    def log_hyperparams(self, params):
        self._task.connect(params)

    def log_metrics(self, metrics, step=None):
        self._task.get_logger().report_scalar(title='Metrics', series=list(metrics.keys())[0], value=list(metrics.values())[0], iteration=step)
    
    def name(self):
        return "ClearMLLogger"