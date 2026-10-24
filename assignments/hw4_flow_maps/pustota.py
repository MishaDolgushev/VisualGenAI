import torch
intervals = torch.linspace(1000, 0, 4).round()
t = torch.LongTensor([668, 320, 950, 10050])
incs =  (torch.searchsorted(intervals, t) -1).clamp(0)
print(intervals)
print(incs)