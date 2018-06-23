import os


file_paths = './Data/'
names = ['BillGates','Tung','LarryPage']
data = {}
for name in names:
    image_dirpath = file_paths + name
    image_filepaths = [os.path.join(image_dirpath, f) for f in os.listdir(image_dirpath)]   
    for i in range(len(image_filepaths)):
        data[image_filepaths[i]] = {'label' : name}
print(data)
