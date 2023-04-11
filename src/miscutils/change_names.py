import os
import sys

'''
This script uniformly renames the files in a specified dir_path. 
'''

def change(dir_path):
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(('.img', '.jpg', '.jpeg', '.png')):
            pass
        else:
            changed = filename[:4]
            os.rename(dir_path + filename, dir_path + changed)


if __name__ == "__main__":
    target_dir = sys.argv[1]
    dir_path = os.getcwd() + '/' + f'{target_dir}/'
    change(dir_path)

