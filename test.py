import os
def read_all_face():
    for root, dirs, files in os.walk('face'):
        print('root_dir:', root)  #当前路径
        print('sub_dirs:', dirs)   #子文件夹
        print('files:', files)     #文件名称，返回list类型
    return files
file_name = read_all_face()