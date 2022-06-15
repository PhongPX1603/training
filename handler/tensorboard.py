from torch.utils.tensorboard import SummaryWriter

class TensorBoard:
    def __init__(self, tb_dir: str):
        self.writer = SummaryWriter(tb_dir)

    def write(self, name: str, value: float, step: int = 0) -> None:
        self.writer.add_scalar(name, value, global_step=step)
