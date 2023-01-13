import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os


#Function to check image files
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class LoadData(Dataset):
    def __init__(self, total_class, path):
        super(LoadData, self).__init__()
        self.total_class = total_class
        self.path = path
        self.directs = []
        self.directs.extend(self.get_img("dog"))
        self.directs.extend(self.get_img("cat"))

    # Function to an encoded data (img, label => 2D tensor, 1D tensor)
    def __getitem__(self, index):
        if "dog" in self.directs[index]:
            x = 0
        else:
            x = 1
        return self.img_to_tensor(self.directs[index]), self.one_hot_vector(x)

    def __len__(self):
        return len(self.directs)
        
    # Function to encode label
    def one_hot_vector(self, index):
        ohv = torch.ones(self.total_class) * 0
        ohv[index] = 1
        return ohv

    # Function to resize image and convert to tensor
    def img_to_tensor(self, img):
        im = Image.open(img)
        resized_img = im.resize((224,224))
        transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        )

        tensor_img = transformer(resized_img)
        return tensor_img

    # Input a path & class name and output all image paths
    def get_img(self, classname):
        sub_path = os.listdir(self.path)
        links = []
        for i in sub_path:
            if classname == i:
                paths = os.listdir(self.path + "/" + i)
                for j in paths:
                    j = self.path + "/" + classname + "/" + j
                    links.append(j)
        return links