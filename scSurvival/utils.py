import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset, TensorDataset

class GPUDataLoaderv0:
    def __init__(self, data, batch_size, shuffle=True, drop_last=False):
        """
        Args:
            data (Tensor or tuple/list of Tensors): Input data stored on GPU.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at the beginning of each epoch.
            drop_last (bool): Whether to drop the last batch if it's smaller than batch_size.
        """
        if isinstance(data, torch.Tensor):  
            self.data = (data,)  # Convert single Tensor to tuple for uniform handling
        elif isinstance(data, (tuple, list)):  
            self.data = tuple(data)  # Convert list to tuple
        else:
            raise TypeError("Data must be a Tensor or a tuple of Tensors!")

        # Ensure all tensors are on GPU
        for d in self.data:
            assert d.is_cuda, "All tensors must be on GPU!"

        self.batch_size = batch_size
        # print(f"Batch size: {self.batch_size}")
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = self.data[0].shape[0]  # Use the first tensor's sample count
        self.device = self.data[0].device

        # if generator is None:
        #     self.generator = torch.Generator(device='cpu')
        #     self.generator.manual_seed(42)
        # else:
        #     self.generator = generator

    def __iter__(self):
        """Generate batch indices at the start of each epoch."""
        if self.shuffle:
            self.indices = torch.randperm(self.num_samples, device=self.device) 
        else:
            self.indices = torch.arange(self.num_samples, device=self.device)

        self.current_batch = 0
        self.num_batches = self.num_samples // self.batch_size if self.drop_last else (self.num_samples + self.batch_size - 1) // self.batch_size
        return self

    def __next__(self):
        """Retrieve the next batch of data."""
        if self.current_batch >= self.num_batches:
            raise StopIteration  # End of iteration

        start = self.current_batch * self.batch_size
        end = min(start + self.batch_size, self.num_samples)

        # print(f"Batch {self.current_batch + 1}/{self.num_batches}, Start: {start}, End: {end}")

        batch_indices = self.indices[start:end]  # Select batch indices on GPU
        batch = tuple(d[batch_indices] for d in self.data)  # Fetch mini-batch from all tensors

        # print(f"Batch shape: {[b.shape for b in batch]}")
        self.current_batch += 1
        return batch

    def __len__(self):
        return self.num_batches


class GPUDataLoaderv1:
    def __init__(self, data, batch_size, shuffle=True, drop_last=False):
        """
        Args:
            data (Tensor or tuple/list of Tensors): Input data stored on GPU.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at the beginning of each epoch.
            drop_last (bool): Whether to drop the last batch if it's smaller than batch_size.
        """
        if isinstance(data, torch.Tensor):  
            self.data = (data,)  # Convert single Tensor to tuple for uniform handling
        elif isinstance(data, (tuple, list)):  
            self.data = tuple(data)  # Convert list to tuple
        else:
            raise TypeError("Data must be a Tensor or a tuple of Tensors!")

        # Ensure all tensors are on GPU
        for d in self.data:
            assert d.is_cuda, "All tensors must be on GPU!"

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = self.data[0].shape[0]  # Use the first tensor's sample count
        self.device = self.data[0].device

    def __iter__(self):
        """Generate batch indices at the start of each epoch."""
        if self.shuffle:
            # Step 1: 全局打乱索引
            self.indices = torch.randperm(self.num_samples, device=self.device)
        else:
            self.indices = torch.arange(self.num_samples, device=self.device)

        # 计算 batch 总数
        self.num_batches = self.num_samples // self.batch_size if self.drop_last else (self.num_samples + self.batch_size - 1) // self.batch_size

        # Step 2: 按 batch 进行分组
        batch_indices = [self.indices[i * self.batch_size: (i + 1) * self.batch_size] for i in range(self.num_batches)]
        
        # Step 3: 再次打乱 batch 顺序
        if self.shuffle:
            batch_indices = [batch_indices[i] for i in torch.randperm(len(batch_indices), device=self.device)]

        # 记录 batch 列表
        self.batches = batch_indices
        self.current_batch = 0
        return self

    def __next__(self):
        """Retrieve the next batch of data."""
        if self.current_batch >= self.num_batches:
            raise StopIteration  # End of iteration

        batch_indices = self.batches[self.current_batch]  # 取当前 batch 的索引
        batch = tuple(d[batch_indices] for d in self.data)  # Fetch mini-batch from all tensors

        self.current_batch += 1
        return batch

    def __len__(self):
        return self.num_batches

    
class IndexDataset(TensorDataset):
    def __init__(self, *tensors):
        super(IndexDataset, self).__init__(*tensors)

    def __getitem__(self, index):
        return index
    
class GPUDataLoader:
    def __init__(self, data, batch_size=128, shuffle=True, num_workers=0, drop_last=False):
        """
        data: Tensor, 已经在 GPU 上的数据
        labels: Tensor, 已经在 GPU 上的标签
        batch_size: int, 每个批次的大小
        shuffle: bool, 是否打乱数据
        num_workers: int, DataLoader 生成索引的工作进程数
        """
        if isinstance(data, torch.Tensor):  
            self.data = (data,)  # Convert single Tensor to tuple for uniform handling
        elif isinstance(data, (tuple, list)):  
            self.data = tuple(data)  # Convert list to tuple
        else:
            raise TypeError("Data must be a Tensor or a tuple of Tensors!")

        # Ensure all tensors are on GPU
        for d in self.data:
            assert d.is_cuda, "All tensors must be on GPU!"

        self.batch_size = batch_size
        self.shuffle = shuffle

        # 用于生成索引的 DataLoader
        self.index_loader = DataLoader(
            IndexDataset(self.data[0]), 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            drop_last=drop_last
        )
        # self.iterator = iter(self.index_loader)
        self.device = self.data[0].device

    def __iter__(self):
        self.iterator = self.index_loader.__iter__()
        return self

    def __next__(self):
        try:
            batch_indices = next(self.iterator).to(self.device)
        except StopIteration:
            raise StopIteration
        
        batch = tuple(d[batch_indices] for d in self.data)
        return batch

    def __len__(self):
        return len(self.index_loader)

