from models import Resnet18Model
from data import LoadData
import torch

# Define gloal variable
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)
EPOCH = 5
SAVED_MODEL_NAME = "./model.pth"

from torch.utils.data import DataLoader

def train():
    # model = VGGModel()
    model = Resnet18Model()
    #khai báo instance của class LoadData
    train_data = LoadData(2, "./data/train")
    test_data = LoadData(2, "./data/test")


    # LoadData trên các instance => kiểu list
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # boo = 1500
    # bar = 15000
    # print("Links of test data: ", len(test_data.directs))
    # print("Links of train data: ", len(train_data.directs))
    # for i in range(100):
    #     print(train_data.directs[i])
    #     print(train_data.__getitem__(i))


    # define kiểu optimize của model
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # so sánh loss xác suất
    loss_fn = torch.nn.CrossEntropyLoss()

    # transfer to GPU
    model = model.to(DEVICE)
    # optimizer = optimizer.to(DEVICE)
    loss_fn = loss_fn.to(DEVICE)

    # train loop
    for i in range(EPOCH):
        # train and cập nhật trọng số
        print("EPOCH:", i)
        model.train()
        for idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, torch.max(target, 1)[1])
            # lan truyền ngược, tính toán hướng cập nhật trọng số
            loss.backward()
            # chạy lr theo hướng đã tính theo loss.backward()
            optimizer.step()
            # print("Output:", output)
            # print("Target:", target)
            
            if idx % 50 == 0:
                print("Batch id:" ,idx, "| Loss:", loss.item())

        # valid, ko update trọng số
        model.eval()
        correct = 0
        for idx, (data, target) in enumerate(test_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)


            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        print(str(correct) + '/' + str(len(test_loader.dataset)))


        # torch.save(model.state_dict(), SAVED_MODEL_NAME)

train()