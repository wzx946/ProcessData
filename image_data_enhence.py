# 使用albumentations数据增强工具对图片进行处理，是一个离线的工作。
import os
from tqdm import tqdm
import cv2
import cv2 as cv
import numpy as np
import albumentations as A
import random
from PIL import Image

def imread(image):
    # image = cv.imread(image)
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # 功能：函数cvCvtColor实现色彩空间转换
    image = image.astype(np.uint8)
    return np.array(image)


# 对单张图片进行数据的增强。
def enhance_oneimage(rootImageName, saveImagePath):
    try:
        image = cv2.imread(rootImageName)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        pil_image = Image.open(rootImageName)
        array_image = np.array(pil_image)
        if array_image.ndim == 3:
            image = cv2.cvtColor(array_image, cv2.COLOR_RGB2BGR)  # 从PIL转换为BGR
        else:
            return 0
    # cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # 加一个判断图片大小的函数。后面的模糊处理操作会收到尺寸大小的影响。
    if image.shape[1] < 50:
        return

    imageDataList = []
    imageNameList = []

    # 图片过暗， RGB-》HSV空间 H:色调， S：饱和度， V：亮度
    # opencv库里的 cv2.convertScaleAbs(image, result, alpha, beta)
    # 其中image是原图，result是输出，alpha是对比度偏置，bate是亮度偏置。一行解决，运算超快
    # img_t = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # h, s, v = cv.split(img_t)
    # # 增加图像亮度
    # image_bright = np.power(image, 1.1)
    # imageDataList.append(image_bright)
    # imageNameList.append("contrast")
    #
    # # 降低图像亮度
    # #
    # # 降低图像对比度
    # v2 = np.clip(cv.add(2 * v, -160), 0, 255)
    # img2 = np.uint8(cv.merge((h, s, v2)))
    # img2 = cv.cvtColor(img2, cv.COLOR_HSV2RGB)
    # imageDataList.append(img2)
    # imageNameList.append("contrast")

    # # 黑屏
    # # 创建指定宽高、3通道、像素值都为0的图像
    # imageDataList.append(np.zeros((image.shape[0], image.shape[1], 3), np.uint8))
    # imageNameList.append("black")
    # # 白屏
    # img = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    # img_rgb = img.copy()
    # img_rgb[:, :, :] = [255, 255, 255]
    # imageDataList.append(img_rgb)
    # imageNameList.append("white")

    # 摄像头遮挡，设置随机黑色色块，至少大于1/3

    # 网络丢包，造成线条等错误。横线和竖线，首先生成多的数量的情况，少量的情况，后续再考虑吧。
    # 1、指定位置；2、指定线条数量； 3、指定线条宽度, 4、随机颜色，不一定要是黑色，设置随机颜色。
    # 指定范围的一维随机浮点数数组
    # LineNums = random.randint(int(image.shape[1] / 4), image.shape[1])  # 线条数量
    # LinePosition = np.random.randint(1, image.shape[1] - 5, LineNums)  # 线条位置。
    # LineWigth = np.random.randint(1, 2, LineNums)  # 线条宽度
    #
    # imageLines = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    # imageLines2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    # for i in range(LineNums):
    #     imageLines[LinePosition[i]:LinePosition[i] + LineWigth[i], :, :] = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
    # #
    # imageDataList.append(imageLines)
    # imageNameList.append("lines")
    #
    # imageLinesBlack = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    # for i in range(LineNums):
    #     imageLinesBlack[LinePosition[i]:LinePosition[i] + LineWigth[i], :, :] = [0, 0, 0]
    # #
    # imageDataList.append(imageLinesBlack)
    # imageNameList.append("linesBlack")
    #
    # # 下部整块数据缺失
    # imageLines2[image.shape[1]-LineNums:image.shape[1], :, :] = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
    # imageDataList.append(imageLines2)
    # imageNameList.append("lines2")
    #
    # imageLines2Black = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    # imageLines2Black[image.shape[1] - LineNums:image.shape[1], :, :] = [0, 0, 0]
    # imageDataList.append(imageLines2Black)
    # imageNameList.append("lines2Black")
    #
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换成RGB颜色空间
    # # 遮挡
    # imageDataList.append(A.RandomSnow(snow_point_upper=0.5, brightness_coeff=3.5, p=1)(image=image)["image"])
    # imageNameList.append("randomSnow")
    # imageDataList.append(A.RandomFog(p=1)(image=image)["image"])
    # imageDataList.append(A.RandomRain(p=1)(image=image)["image"])
    # imageDataList.append(A.RandomSunFlare(num_flare_circles_lower=7, num_flare_circles_upper=20, src_radius=350, p=1)(image=image)["image"])
    # imageNameList.append("randomSunFlare")
    # 模糊
    # imageDataList.append(A.Blur(blur_limit=(200, 300), p=1.0)(image=image)["image"])  # 因为我们是做的数据增强，因此概率p还是100%比较好。
    # imageNameList.append("gaussianBlur")
    # # 运动模糊， 模拟摄像头转动
    # imageDataList.append(A.MotionBlur(blur_limit=(200, 300), p=1)(image=image)["image"])
    # imageNameList.append("motionBlur")
    # 加噪
    # imageDataList.append(A.GaussNoise(var_limit=(200, 300.0), mean=5, always_apply=True)(image=image)["image"])
    # imageNameList.append("noise")
    # imageDataList.append(A.ISONoise(color_shift=(0.5, 1), intensity=(10, 20), always_apply=True)(image=image)["image"])
    # imageNameList.append("noise2")

    # # 偏色，通道丢弃
    # imageDataList.append(A.ChannelDropout(channel_drop_range=(1, 1), always_apply=True)(image=image)["image"])
    # imageNameList.append("remove")
    # # 通道置乱
    # imageDataList.append(A.ChannelShuffle(always_apply=True)(image=image)["image"])
    # imageNameList.append("disorder")
    # # 图片超像素化
    # # A.Superpixels()
    # imageDataList.append(A.Superpixels(p=1, p_replace=0.5)(image=image)["image"])
    # imageNameList.append("superpixels")
    # 图片压缩
    # imageDataList.append(A.ImageCompression(quality_lower=1, quality_upper=2, always_apply=True)(image=image)["image"])
    # imageNameList.append("compression")
    # # 图片反转
    # imageDataList.append(A.InvertImg(always_apply=True)(image=image)["image"])
    # imageNameList.append("reversal")
    # # 透射变换
    # imageDataList.append(A.Perspective(scale=(0.05, 0.1), keep_size=True, always_apply=True)(image=image)["image"])
    # imageNameList.append("transmission")
    # # 图片黑白化
    # imageDataList.append(A.to_gray(image))
    # imageNameList.append("gray")

    # 随机雨雪
    imageDataList.append(A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.1, alpha_coef=0.08, always_apply=False, p=0.5)(image=image)["image"])
    imageNameList.append("RandomFog")

    # imageDataList.append(A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7,
    #            brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.5)(image=image)["image"])
    # imageNameList.append("RandomRain")

    # imageDataList.append(
    #     A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=1)(image=image)[
    #         "image"])
    # imageNameList.append("RandomSnow")
    
    # imageDataList.append(A.RandomSunFlare(flare_roi=(0, 0, 0.9, 0.9), angle_lower=None, angle_upper=None, num_flare_circles_lower=None,
    #                                       num_flare_circles_upper=None, src_radius=200, src_color=(255, 255, 255),
    #                                       angle_range=(0, 1), num_flare_circles_range=(6, 10), always_apply=None, p=1)(image=image)[
    #         "image"])
    # imageNameList.append("RandomSunFlare")
    # imageDataList.append(A.HorizontalFlip(p=1)(image=image)["image"])
    # imageDataList.append(A.OpticalDistortion(p=1)(image=image)["image"])  # 光学畸变
    # imageDataList.append(A.GridDistortion(p=1)(image=image)["image"])  # 网格失真

    for index in range(len(imageDataList)):
        image_name_o = imageNameList[index] + '.jpg'
        imageName = saveImagePath + '_' + image_name_o
        
        # img = cv2.cvtColor(imageDataList[index], cv2.COLOR_RGB2BGR)  # 需要转会到opencv要求的颜色空间去，从RGB --> BGR，
        # cv2.imwrite(imageName, img)  # 在后面加上一个何种图片操作的名字。
        try:   # 与其想那么多，还不如直接加一个判断语句。
            img = cv2.cvtColor(imageDataList[index], cv2.COLOR_RGB2BGR)  # 需要转会到opencv要求的颜色空间去，从RGB --> BGR，
            cv2.imwrite(imageName, img)  # 在后面加上一个何种图片操作的名字。
        except:
            continue


if __name__ == '__main__':
    # 这种是组合变换，对单幅图像进行同时进行多种操作。
    # transform = A.Compose([
    #     A.RandomCrop(width=256, height=256),
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.2),
    # ])
    import time
    import eventlet  # 导入eventlet这个模块

    eventlet.monkey_patch()  # 必须加这条代码 主要是为了调用一些简写的路劲，先定位一个相对路径地址。

    # tmpImagePath = '/root/cvdata/lena.jpg'
    # saveImagePath = '/root/cvdata'
    # enhance_oneimage(tmpImagePath, saveImagePath)

    imageRootPath = [r"F:\workfile\20240813data\20240816knife_stick_v2\val"]
    # imageRootPath = ["/root/cvdata/workFile/datasets/distortion/tmp"]

    # 原文解释，其实就是一个是RGB还是BGR的问题，避免读错。
    # OpenCV reads an image in BGR format (so color channels of the image have the following order: Blue, Green, Red).
    # Albumentations uses the most common and popular RGB image format.
    # So when using OpenCV, we need to convert the image format to RGB explicitly.
    # for index in imageRootPath:
    index = r"F:\workfile\gudong\20240813data\20240816knife_stick_v2\test"
    imageList = os.listdir(index)

    # 保存图片的路径
    saveRootPath = index+'enhance'
    if not os.path.exists(saveRootPath):
        os.makedirs(saveRootPath)
    for imageName in tqdm(imageList):
        if imageName.endswith('.jpg') or imageName.endswith('.png'):
            imagePath = os.path.join(index, imageName)
            saveImagePath = os.path.join(saveRootPath, imageName[:-4])
            enhance_oneimage(imagePath, saveImagePath)
        else:
            continue
# image2 = A.RandomSnow(p=0.5)(image=image)["image"]  #
# 上述保存了多幅图像，如何进行一个保存？ 不是多幅图像，是进行多种操作，一张图片进行了很多次的操作。
