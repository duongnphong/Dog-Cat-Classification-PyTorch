from torch.utils.data import DataLoader
from models import Resnet18Model
from data import LoadData
import torch

# Define global variable
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EPOCH = 10
SAVED_MODEL_NAME = "./model.pth"

def train(): 
    # model = VGGModel()
    model = Resnet18Model()
    # instance of class LoadData
    train_data = LoadData(2, "./data/train")
    test_data = LoadData(2, "./data/test")

    # LoadData trên các instance => kiểu list
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # define kiểu optimize của model
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # transfer to GPU
    model = model.to(DEVICE)
    # optimizer = optimizer.to(DEVICE)
    loss_fn = loss_fn.to(DEVICE)

    # train loop
    for i in range(EPOCH):
        # train and update parameters
        print("EPOCH:", i)
        model.train()
        for idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, torch.max(target, 1)[1])
            # backpropagation, caclulate direction to update parameters
            loss.backward()
            # apply lr according to the direction from loss.backward()
            optimizer.step()


            if idx % 50 == 0:
                print("Batch id:", idx, "| Loss:", loss.item())

        # valid loop
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

        torch.save(model.state_dict(), SAVED_MODEL_NAME)

train()
