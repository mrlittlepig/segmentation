import shutil
import cv2
import os
import create_synthetic_imgs

# get file list in source dir
def get_file_list(src_dir):
    file_list = []
    for filename in os.listdir(src_dir):
        file_list.append(filename)
        # print(filename)
    return file_list

# copy source files paths in plane txt file to target dir
def copy_files(src_dir, dst_dir):
    file_index = 0
    file_list = get_file_list(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for file in file_list:
        tmp_file_path = src_dir + file
        dst_file_path = dst_dir + file
        if os.path.isfile(tmp_file_path):
            shutil.copyfile(tmp_file_path, dst_file_path)
            # print(dst_file_path)
            file_index += 1
    # print(file_index)

def rename_images(img_dir, dst_dir, bak_end = "jpg"):
    image_list = get_file_list(img_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    count = 0
    for image in image_list:
        count += 1
        image_path = os.path.join(img_dir, image)
        img = cv2.imread(image_path)
        dst_path = os.path.join(dst_dir,("%05d." % count)+bak_end)
        cv2.imwrite(dst_path, img)
        print dst_path

def read_image_mat(src_img):
    img = cv2.imread(src_img)
    return img

# copy souce dir to dst_dir
def copy_dir_files(src_dir, dst_dir, filter=True):
    file_list = get_file_list(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for file in file_list:
        tmp_file_path = src_dir + file
        dst_file_path = dst_dir + file
        if os.path.isfile(tmp_file_path):
            shutil.copyfile(tmp_file_path, dst_file_path)
            # print(dst_file_path)
        if os.path.isdir(tmp_file_path):
            # filter to skip file or folder like ".*"
            # print(tmp_file_path)
            if filter == True:
                if file[0] == '.':
                    continue
                copy_dir_files(tmp_file_path + '/', dst_file_path + '/', filter=True)

def copy_one_file(src_file, dst_file):
    shutil.copyfile(src_file, dst_file)

def move_one_file(src_file, dst_file):
    shutil.move(src_file, dst_file)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def isfile(path):
    if not os.path.exists(path):
        print("path not exists: %s" % path)
        return 0
    if os.path.isdir(path):
        print("path is dir: %s" % path)
        return 0
    if path.split("/")[-1][0] == ".":
        print("path is been hided %s" % path)
        return 0
    return 1

if __name__ == "__main__":
    img_dir = "last_images"
    dst_dir = "last_rename"
    # rename_images(img_dir, dst_dir, bak_end="jpg")
