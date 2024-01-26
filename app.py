import torch
import torch.nn as nn
import torch.nn.functional as F
# import model
# from model import imodel

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.in_channels = in_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or self.in_channels != out.shape[1]:
            identity = F.avg_pool2d(identity, kernel_size=self.stride, stride=self.stride)
            identity = F.pad(identity, (0, 0, 0, 0, 0, out.shape[1] - self.in_channels), "constant", 0.0)

        out += identity
        out = self.relu(out)

        return out



class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.in_channels = 64  # Initial number of channels
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, self.in_channels, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, planes, stride))  # Pass stride for the first block
        self.in_channels = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x



# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

# U-Net Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = ResNetEncoder()
        self.decoder = nn.ModuleList([
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 16)
        ])
        self.segmentation_head = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        for decoder_block in self.decoder:
            x = decoder_block(x)
        x = self.segmentation_head(x)
        x = self.sig(x)
        return x



device = torch.device("cpu")
# imodel = UNet()
# imodel.load_state_dict(torch.load('/home/team3s/thesis/model_name.pth', map_location=device))
# imodel.eval()
#------------------------------
# model = torch.jit.load('/home/team3s/thesis/final_model-Sami.pt')
# model.eval()
    #-------------


#------------
from flask import Flask
from flask_cors import CORS
import numpy as np
from flask import Flask, request, jsonify
import base64
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)



test_transforms = A.Compose([
    A.Resize(224, 224),
], is_check_shapes=False)



# model = torch.jit.load('/home/team3s/thesis/final_model-Sami.pt')
# model.eval()
# model = torch.jit.load("/home/team3s/thesis/final_model-Sami.pt")




pth_file_path = '/home/team3s/thesis/final_model-Sami.pt'


import mimetypes


# Determine the file type
file_extension, _ = mimetypes.guess_type(pth_file_path)





@app.route('/add', methods=['POST'])
def add_numbers():
    try:
        data = request.get_json()
        num1 = data['num1']
        num2 = data['num2']
        result = num1 + num2
        return jsonify({'sum': result})
    except KeyError:
        return jsonify({'error': 'Please provide both num1 and num2 as JSON in the request body.'}), 400


@app.route('/predict', methods=['POST'])
def predict():
  try:
    image_file = request.files['image']
    image_path = '/home/team3s/thesis/temp_image.jpg'
    # image_path = 'temp_image.jpg'
    image_file.save(image_path)
    image = Image.open(image_path).convert('L')
    image = np.expand_dims(image, axis=-1)
    device = torch.device("cpu")
    test_transforms = A.Compose([
        A.Resize(224, 224),
    ], is_check_shapes=False)

    image = test_transforms(image=image)['image']
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    imageTen = torch.Tensor(image) / 255.0



    # with torch.no_grad():
    # # #     imodel.eval()
    # # #   outputs = imodel.forward(imageTen.unsqueeze(0))
    #   imageTen = imageTen.to(device)
    #   model = model.to(device)
    #   outputs = model(imageTen.unsqueeze(0))


    #   threshold = 0.5
    #   binary_mask = (outputs > threshold).float()
    #   # binary_mask_np = binary_mask.cpu().numpy().squeeze()
    #   binary_mask_np = binary_mask.squeeze()


    #   # overlay = np.zeros_like(image.cpu().numpy().squeeze(), dtype=np.uint8)
    #   overlay = np.zeros_like(image.squeeze(), dtype=np.uint8)
    #   overlay[binary_mask_np > 0.5] = 255

    #   plt.imshow(image.squeeze(), alpha=1, cmap='gray')
    #   plt.imshow(overlay, cmap='gray', alpha=0.5)

    #   overlay_path = "/home/team3s/thesis/overlay_image.png"
    #   plt.axis('off')
    #   plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
    #   plt.close()

    # with open('/home/team3s/thesis/overlay_image.png', "rb") as img_file:
    #         encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
    # return jsonify({'result': "Successful", 'image_data': encoded_image})


    return jsonify({'result': "Successuiy"})
  except Exception as e:
    # print(f"Error: {str(e)}")
    return jsonify({"result-error": str(e)})


@app.route('/')
def hello():
    return str(file_extension)
    # return 'Hello, Bye dddwgBwhhyed'

if __name__ == '__main__':
    app.run()  # Use a port number that is not in use
