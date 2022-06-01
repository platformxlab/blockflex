import sys
import albumentations as A
import cv2
import os
import threading
threads = 24
#Getting the FS behavior of ml image prep

DIR="/home/breidys2/benchmarks/ml_prep/images/Data/DET/train/images"
DIR="/mnt/nvme0n1/ml_prep/input"
DIR_1="/mnt/nvme0n1/ml_prep/output/1"
DIR_2="/mnt/nvme0n1/ml_prep/output/2"
DIR_3="/mnt/nvme0n1/ml_prep/output/3"

#Declare image augmentation pipeline
trans_1 = A.Compose([
   # A.RandomCrop(width=64,height=64),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
trans_2 = A.Compose([
    A.RandomCrop(width=128,height=128),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])
trans_3 = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
def process(part):
    for img in part:
        #Read an image with OpenCV and convert it to the RGB colorspace
        #image = cv2.imread("image.jpg")
        #print(DIR+"/"+img)
        image = cv2.imread(DIR+"/"+img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Augment the image
        try:
            trans_img = trans_1(image=image)["image"]
            cv2.imwrite(f"{DIR_1}/{img[:-5]}_1.JPEG", trans_img)
        except ValueError:
            pass
        try:
            trans_img = trans_2(image=image)["image"]
            cv2.imwrite(f"{DIR_2}/{img[:-5]}_2.JPEG", trans_img)
        except ValueError:
            pass
        try:
            trans_img = trans_3(image=image)["image"]
            cv2.imwrite(f"{DIR_3}/{img[:-5]}_3.JPEG", trans_img)
        except ValueError:
            pass
    
if __name__ == "__main__":
    wait_list = list()
    imgs = os.listdir(DIR)
    cut = len(imgs)//threads
    for i in range(threads):
        x = threading.Thread(target=process, args=(imgs[cut*i:cut*(i+1)],))
        x.start()
        wait_list.append(x)

    for i, thread in enumerate(wait_list):
        thread.join()
        print(str(i) + " Done...")





