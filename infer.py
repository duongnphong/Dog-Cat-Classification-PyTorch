import urllib.request
import torch
from models import Resnet18Model
from torchvision import transforms
from PIL import Image
import cv2

SAVED_MODEL_NAME = "./model.pth"

# Load saved model by creating an instance of the model class
model = Resnet18Model()
model.load_state_dict(torch.load(SAVED_MODEL_NAME))
model.eval()

def main():
    # img = img_to_tensor("https://i.kym-cdn.com/photos/images/newsfeed/001/741/229/1d0.jpg").unsqueeze(0)
    img = img_to_tensor("https://znews-photo.zingcdn.me/w660/Uploaded/aeixrdbkxq/2021_02_22/a5.jpg").unsqueeze(0)

    # print(img.size())
    output = model(img)

    pred = torch.argmax(output, dim=1)
    conf = output[0][pred]

    image = cv2.imread("assets/infer.jpg")
    window_name = "Result"
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20,30)
    fontScale = 1
    color = (0,0,0)
    thickness = 2
    x = round(float(conf)*100, 2)
    if pred == 0:
        image = cv2.putText(image, f'DOG {x}', org, font, fontScale, color, thickness)
    else:
        image = cv2.putText(image, f'CAT {x}', org, font, fontScale, color, thickness)


    cv2.imwrite("assets/infer_1.jpg", image)

    # cv2.imshow(window_name, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def img_to_tensor(img):
    urllib.request.urlretrieve(img, "assets/infer.jpg")
    im = Image.open("infer.jpg")
    # im.show()
    resized_img = im.resize((224,224))
    transformer = transforms.ToTensor()
    tensor_img = transformer(resized_img)
    return tensor_img

if __name__ == "__main__":
    main()