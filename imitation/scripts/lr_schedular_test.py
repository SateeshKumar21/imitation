import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

model = MyModel()
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=500*1000, eta_min=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=1, T_mult=1, eta_min=1e-5)

all_lr = []
for i in tqdm(range(500*1000)):
    optim.zero_grad()
    # loss = model(torch.randn(10)).sum()
    # loss.backward()
    optim.step()
    scheduler.step()
    
    all_lr.append(scheduler.get_last_lr()[0])

plt.plot(all_lr)
plt.show()