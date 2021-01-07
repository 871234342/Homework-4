import argparse
import math
import os
import random
import numpy as np

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.cuda import amp
from tqdm import tqdm
from PIL import Image

from vdsr_pytorch import DatasetFromFolder
from vdsr_pytorch import VDSR


def add_noise(img):
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)

    L = random.random() / 5
    img = img * (1 - L) + L * torch.randn(img.shape)

    to_PIL = transforms.ToPILImage()
    img = to_PIL(img)
    return img


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
num_worker = 4
num_epochs = 1500
image_size = 256
batch_size = 32
lr = 0.05
momentum = 0.9
weight_decay = 0.0001
clip = 0.4
scale_factor = 3
print_frequency = 5
manual_seed = 0
crop_size = image_size - (image_size % scale_factor)

try:
    os.makedirs("weights")
except OSError:
    pass

random.seed(manual_seed)
torch.manual_seed(manual_seed)

cudnn.benchmark = True

transform = transforms.Compose([
    transforms.RandomCrop(image_size, pad_if_needed=True),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.Lambda(lambda x: add_noise(x))
])

train_dataset = DatasetFromFolder("training_hr_images",
                                  image_size=image_size,
                                  scale_factor=scale_factor,
                                  transform=transform)
val_dataset = DatasetFromFolder("val_hr_images",
                                image_size=image_size,
                                scale_factor=scale_factor)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=int(num_worker))
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=int(num_worker))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VDSR().to(device)

criterion = nn.MSELoss().to(device)
# init lr: 0.1 decay 0.00001, so decay step 5.
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 5, gamma=0.1)

best_psnr = 0.
best_epoch = 0

# Creates a GradScaler once at the beginning of training.
scaler = amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for iteration, (inputs, target) in progress_bar:
        optimizer.zero_grad()

        inputs, target = inputs.to(device), target.to(device)

        # Runs the forward pass with autocasting.
        with amp.autocast():
            output = model(inputs)
            loss = criterion(output, target)
        # Scales loss.  Calls backward() on scaled loss to
        # create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose
        # for corresponding forward ops.
        scaler.scale(loss).backward()

        # Adjustable Gradient Clipping.
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        # scaler.step() first unscales the gradients of
        # the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs,
        # optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        progress_bar.set_description(f"[{epoch + 1}/{num_epochs}][{iteration + 1}/{len(train_dataloader)}] "
                                     f"Loss: {loss.item():.6f} ")

    # Test
    model.eval()
    avg_psnr = 0.
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for iteration, (inputs, target) in progress_bar:
            inputs, target = inputs.to(device), target.to(device)

            prediction = model(inputs)
            mse = criterion(prediction, target)
            psnr = 10 * math.log10(1 / mse.item())
            avg_psnr += psnr
            progress_bar.set_description(f"Epoch: {epoch + 1} [{iteration + 1}/{len(val_dataloader)}] "
                                         f"Loss: {mse.item():.6f} "
                                         f"PSNR: {psnr:.5f}.")

    # Dynamic adjustment of learning rate.
    scheduler.step()

    # Save model
    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f"weights/vdsr_{scale_factor}x_epoch_{epoch + 1}.pth")
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), f"weights/vdsr_{scale_factor}x_best.pth")
        best_epoch = epoch

    print(f"Average PSNR: {avg_psnr / len(val_dataloader):.5f} dB.  ",
          f"Best so far: {best_psnr:.5f} dB at {best_epoch} epoch.")

print(f'Summary: best_psnr: {best_psnr:.5f} at epoch {best_epoch}')

# Inference
file_folder = "testing_lr_images/"
model.load_state_dict(torch.load(f"weights/vdsr_{scale_factor}x_best.pth",
                                 map_location=device))
model.eval()

# Create output dir
output_folder = "testing_hr_images"
if not os.path.isdir(output_folder):
    os.mkdir("testing_hr_images")

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
    image = image.resize((image_width, image_height), Image.BILINEAR)

    preprocess = transforms.ToTensor()
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


