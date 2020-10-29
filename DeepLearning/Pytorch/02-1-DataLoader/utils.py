import os, random, shutil, glob

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def split_dataset(dataset_dir, split_dir):
    """划分数据为训练集，验证集和测试集"""
    random.seed(1)
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs:
            imgs = glob.glob(os.path.join(root, sub_dir, '*.jpg'))
            random.shuffle(imgs)

            img_count = len(imgs)
            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))
            
            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                src = imgs[i]
                dst = os.path.join(out_dir,imgs[i].split('/')[-1])
                shutil.copy(src, dst)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, int(len(imgs)*train_pct), valid_point-train_point, img_count-valid_point))