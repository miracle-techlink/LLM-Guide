# data_prefetcher.py

import torch
from torch.utils.data import DataLoader

class DataPrefetcher:
    """异步预取数据加载器到 *device*（第一个管道阶段）。"""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._prefetch()

    def _prefetch(self):
        """预取下一个输入和目标数据。"""
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        """返回下一个输入和目标数据。"""
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        if self.next_input is None:
            raise StopIteration
        input_, target = self.next_input, self.next_target
        self._prefetch()
        return input_, target