import argparse
import os

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image

from vdsr_pytorch import VDSR

parser = argparse.ArgumentParser(description="Accurate Image Super-Resolution Using Very Deep Convolutional Networks")
parser.add_argument("--file", type=str, default="./assets/baby.png",
                    help="Test low resolution image name. (default:`./assets/baby.png`)")
parser.add_argument("--scale-factor", default=4, type=int, choices=[2, 3, 4],
                    help="Super resolution upscale factor. (default:4)")
parser.add_argument("--weights", type=str, default="weights/vdsr_4x.pth",
                    help="Generator model name.  (default:`weights/vdsr_4x.pth`)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")

file_folder = "testing_lr_images/"
scale_factor = 3
weights_path = "weights/vdsr_3x_best.pth"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

args = parser.parse_args()

cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model
model = VDSR().to(device)

# Load state dicts
model.load_state_dict(torch.load(weights_path, map_location=device))

# Set eval mode
model.eval()

# Create output dir
output_folder = "testing_hr_images"
if not os.path.isdir(output_folder):
    os.mkdir("testing_hr_images")

# Open image

for image_file in os.listdir(file_folder):
    path = file_folder + image_file
    # 1-channel
    '''#print("Dealing with image: ", image_file)

    image = Image.open(path).convert("YCbCr")
    image_width = int(image.size[0] * scale_factor)
    image_height = int(image.size[1] * scale_factor)
    image = image.resize((image_width, image_height), Image.BICUBIC)
    y, cb, cr = image.split()

    preprocess = transforms.ToTensor()
    inputs = preprocess(y).view(1, -1, y.size[1], y.size[0])

    inputs = inputs.to(device)

    with torch.no_grad():
        out = model(inputs)
    out = out.cpu()
    out_image_y = out[0].detach().numpy()
    out_image_y *= 255.0
    out_image_y = out_image_y.clip(0, 255)
    out_image_y = Image.fromarray(np.uint8(out_image_y[0]), mode="L")

    out_img_cb = cb.resize(out_image_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_image_y.size, Image.BICUBIC)
    out_img = Image.merge("YCbCr", [out_image_y, out_img_cb, out_img_cr]).convert("RGB")'''

    # 3-channel
    image = Image.open(path).convert("RGB")
    image_width = int(image.size[0] * scale_factor)
    image_height = int(image.size[1] * scale_factor)
    #image = image.resize((image_width, image_height), Image.BILINEAR)

    preprocess = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
    ])
    inputs = preprocess(image)
    inputs = inputs.unsqueeze(0)
    inputs = inputs.to(device)

    with torch.no_grad():
        out = model(inputs)
    out = out.cpu()
    out_image = out[0].detach().numpy()
    out_image *= 255.0
    out_image = out_image.astype(np.uint8)
    out_image = np.transpose(out_image, (1, 2, 0))
    out_img = Image.fromarray(out_image, mode="RGB")

    # save image
    path = "testing_hr_images/" + image_file
    out_img.save(path)

