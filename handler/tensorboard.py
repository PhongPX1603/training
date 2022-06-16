from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorBoard:
    def __init__(self, tb_dir: str = None):
        super(TensorBoard, self).__init__()
        time = datetime.now().strftime(r'%y%m%d%H%M')
        self.tb_dir = f'{tb_dir}/{time}'
        self.writer = SummaryWriter(self.tb_dir)

    def write(self, name: str, value: float, step: int = 0) -> None:
        self.writer.add_scalar(name, value, global_step=step)
