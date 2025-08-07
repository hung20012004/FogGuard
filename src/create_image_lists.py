# Tạo file create_image_lists.py
import os
import glob

# Tạo thư mục config
os.makedirs(r'data\VOC\config', exist_ok=True)

# Tạo danh sách cho VOC2007 train
voc2007_train_path = r'data\VOC\train\VOCdevkit\VOC2007\images'
with open(r'data\VOC\config\2007_train.txt', 'w') as f:
    for img in glob.glob(os.path.join(voc2007_train_path, '*.jpg')):
        f.write(os.path.abspath(img) + '\n')

# Tạo danh sách cho VOC2012 train  
voc2012_train_path = r'data\VOC\train\VOCdevkit\VOC2012\images'
with open(r'data\VOC\config\2012_train.txt', 'w') as f:
    for img in glob.glob(os.path.join(voc2012_train_path, '*.jpg')):
        f.write(os.path.abspath(img) + '\n')

# Tạo danh sách cho VOC2007 test
voc2007_test_path = r'data\VOC\test\VOCdevkit\VOC2007\images'  
with open(r'data\VOC\config\2007_test.txt', 'w') as f:
    for img in glob.glob(os.path.join(voc2007_test_path, '*.jpg')):
        f.write(os.path.abspath(img) + '\n')

print("Created image lists successfully!")