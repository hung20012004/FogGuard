# Tạo file combine_files.py
import os

config_dir = "D:\Foggy\FogGuard\data\VOC\config"

# Kết hợp train files
with open(os.path.join(config_dir, 'trainval.txt'), 'w') as outfile:
    with open(os.path.join(config_dir, 'voc2007_train.txt'), 'r') as f1:
        outfile.write(f1.read())
    with open(os.path.join(config_dir, 'voc2012_train.txt'), 'r') as f2:
        outfile.write(f2.read())

# Copy test file  
with open(os.path.join(config_dir, 'test.txt'), 'w') as outfile:
    with open(os.path.join(config_dir, 'voc2007_test.txt'), 'r') as f:
        outfile.write(f.read())

print("Combined files successfully!")