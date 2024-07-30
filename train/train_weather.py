import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


from models.basic.resnet import ResNet, BasicBlock
from data.dataset import DataLoaderTrain, get_training_data
from torch.utils.data import DataLoader
from utils.init_weights import init_xavier_weights
from utils.scheduler import simple_scheduler


##------------------------------------------------------
## SET HYPERPARAMETERS AND DEFAULT CONFIGURATIONS
learning_rate = 0.1
num_epoch = 50
model_name = 'model.pth'
seed = 214
model = ResNet(BasicBlock, [3, 3, 3])
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
BATCH_SIZE = 4

train_loss = 0
valid_loss = 0
correct = 0
total_cnt = 0
best_acc = 0


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
train_dir = './data/weather_train_data'
train_dataset = get_training_data(train_dir)

train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [9500, 500])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)

valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)



torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
# random.seed(0)
##------------------------------------------------------


# Train
for epoch in range(num_epoch):
    print(f"====== { epoch+1} epoch of { num_epoch } ======")
    model.train()
    lr_scheduler = simple_scheduler(optimizer, epoch, learning_rate=learning_rate)
    train_loss = 0
    valid_loss = 0
    correct = 0
    total_cnt = 0
    # Train Phase
    for step, batch in enumerate(train_loader):
        #  input and target
        batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        logits = model(batch[0])
        loss = loss_fn(logits, batch[1])
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predict = logits.max(1)

        total_cnt += batch[1].size(0)
        correct +=  predict.eq(batch[1]).sum().item()

        if step % 100 == 0 and step != 0:
            print(f"\n====== { step } Step of { len(train_loader) } ======")
            print(f"Train Acc : { correct / total_cnt }")
            print(f"Train Loss : { loss.item() / batch[1].size(0) }")

    correct = 0
    total_cnt = 0

# Test Phase
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(valid_loader):
            # input and target
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model(batch[0])
            valid_loss += loss_fn(logits, batch[1])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
        valid_acc = correct / total_cnt
        print(f"\nValid Acc : { valid_acc }")
        print(f"Valid Loss : { valid_loss / total_cnt }")

        if(valid_acc > best_acc):
            best_acc = valid_acc
            # torch.save(model, '/content/drive/MyDrive/ResNet18_weight.pth')
            torch.save(model, f'./train/weight/ResNet18_weight_val.acc_{valid_acc}.pth')
            print("Model Saved!")