import os
import random

# Divide dataset into  train set,  val set, test set
#

label='label/'
saveBasePath="datasets/"

train_percent = 0.7
val_percent = 0.15
test_percent = 0.15

temp_files = os.listdir(label)
total_files = []
for tif_file in temp_files:
    if tif_file.endswith(".tif"):
        total_files.append(tif_file)

num=len(total_files)

list=range(num)

train_size = int(num*train_percent)

val_size = int(num*val_percent)


train_and_val_size = train_size + val_size

train_val_data = random.sample(list, train_and_val_size)

train_data = random.sample(train_val_data, train_size)

print("train and val size", train_and_val_size)
print("train size",train_size)


ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')


for i  in list:
    name=total_files[i][:-4]+'\n'
    if i in train_val_data:
        ftrainval.write(name)
        if i in train_data:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
