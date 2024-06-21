from PIL import Image
from lavis.models import load_model_and_preprocess
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import torch
import numpy as np
# generate caption via blip frame by frame


def get_img_caption_ntu(idex):
    num_sample = 10000000
    # device = torch.device("cuda" if torch.cuda().is_available() else "cpu")
    torch.cuda.set_device(idex+1)
    # use any image caption models. blip, ofa or monkey
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True, device="cuda")

    image_path = '../data/image/'
    save_path = '../data/text/'

    all_img = os.listdir(image_path)
    all_img.sort()

    all_img = all_img[idex*num_sample: (idex+1)*num_sample]

    for sub_file in all_img:
        sub_img_file = os.listdir(image_path + sub_file)
        sub_img_file.sort()

        if not os.path.exists(os.path.join(save_path, sub_file)):
            os.makedirs(os.path.join(save_path, sub_file))
        id = 0
        print(sub_file)
        for img_s in sub_img_file:
            if os.path.isfile(save_path+sub_file+'/' + str(id).zfill(6) + '.txt'):
                continue
            # load sample image
            raw_image = Image.open(os.path.join(
                image_path, sub_file, img_s)).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).cuda()
            text = model.generate({"image": image})

            with open(save_path+sub_file+'/' + str(id).zfill(6) + '.txt', 'w') as f:
                f.write(text[0])
            id = id+1


# generate text with video image
def concate_image(idex, image_path='../data/image/'):
    save_path = '../data/ntu_video_image/'
    num_sample = 20000
    all_img_file = os.listdir(image_path)
    all_img_file.sort()
    all_img_file = all_img_file[idex*num_sample: (idex+1)*num_sample]

    for idd, image_name in enumerate(all_img_file):
        print(idd, image_name, idd/num_sample)
        all_img = []
        names = os.listdir(os.path.join(image_path, image_name))
        names.sort()
        for sub_name in names:
            temp_img = cv2.imread(os.path.join(
                image_path, image_name, sub_name))
            if len(all_img) >= 2 and all_img[-2].shape[0] != temp_img.shape[0]:
                temp_img = cv2.resize(
                    temp_img, (temp_img.shape[1], all_img[-2].shape[0]))

            temp_img[:, 0:3, :] = 255
            temp_img[:, -3:, :] = 255
            temp_img[0:3, :, :] = 255
            temp_img[-3:, :, :] = 255
            all_img.append(temp_img)
        read_cat_img(all_img, image_name, save_path)


def read_cat_img(images, image_name, save_path):
    images_1 = images[0:len(images)-1:2]
    images_2 = images[1:len(images):2]

    images_1 = np.concatenate(images_1, axis=1)
    images_2 = np.concatenate(images_2, axis=1)
    pp = os.path.join(save_path, image_name)
    if not os.path.exists(pp):
        os.makedirs(pp)

    cv2.imwrite(os.path.join(save_path, image_name, '001.jpg'), images_1)
    cv2.imwrite(os.path.join(save_path, image_name, '002.jpg'), images_2)


# concate_image(3, '/data/data2/userdata/tanbo/tanbo/ntu_rgb_mask_61_120/ntu120_rgb_image_60_120/')
def reshape_imge(idex=0):
    save_path = "../data/ntu_video_image_resize_small/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_path = "../data/ntu_video_image/"
    num_sample = 40000
    all_img_file = os.listdir(image_path)
    all_img_file.sort()
    all_img_file = all_img_file[idex*num_sample: (idex+1)*num_sample]

    for idd, image_name in enumerate(all_img_file):
        print(idd, image_name, idd/num_sample)
        all_img = []
        names = os.listdir(os.path.join(image_path, image_name))
        names.sort()
        if not os.path.exists(os.path.join(save_path, image_name)):
            os.makedirs(os.path.join(save_path, image_name))

        for sub_name in names:
            temp_img = cv2.imread(os.path.join(
                image_path, image_name, sub_name))
            higth, width = temp_img.shape[0], temp_img.shape[1]
            w_rate, h_rate = 1, 1
            if temp_img.shape[1] > 448:
                w_rate = 448/temp_img.shape[1]
                width = 448
            if temp_img.shape[0] > 448:
                h_rate = 448/temp_img.shape[0]
                higth = 448
            rates = min(h_rate, w_rate)
            temp_img = cv2.resize(
                temp_img, (int(temp_img.shape[1]*rates), int(temp_img.shape[0]*rates)))
            cv2.imwrite(os.path.join(
                save_path, image_name, sub_name), temp_img)


get_img_caption(0)
