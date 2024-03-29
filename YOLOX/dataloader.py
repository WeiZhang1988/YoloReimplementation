from torch.utils.data import dataloader

class YoloXDataLoader(dataloader.DataLoader):
  count = 0
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
    self.iterator = super().__iter__()
    YoloXDataLoader.count += 1
  def __del__(self):
    YoloXDataLoader.count -= 1
  def __len__(self):
    return len(self.batch_sampler.sampler)
  def __iter__(self):
    for _ in range(len(self)):
      yield next(self.iterator)

class _RepeatSampler:
  def __init__(self, sampler):
    self.sampler = sampler
  def __iter__(self):
    while True:
      yield from iter(self.sampler)