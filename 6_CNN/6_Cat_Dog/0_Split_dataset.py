import wget
import zipfile
import os
import shutil
from PIL import Image, UnidentifiedImageError

base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir,'PetImages')    # Source Directory
dst_dir = os.path.join(base_dir,'data')         # Destination Directory
# Download kaggle Cats and Dogs dataset zip file from download.microsoft.com
# Be careful with license attached on zip file.
url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
zipName = os.path.join(base_dir,'kagglecatsanddogs_3367a.zip')

if not os.path.isdir(src_dir):
    if not os.path.isfile(zipName):
        wget.download(url)
    datazip = zipfile.ZipFile(zipName)
    datazip.extractall(base_dir)
    os.remove('MSR-LA - 3467.docx')
    os.remove('readme[1].txt')
    #os.remove(zipName)

src_cat_dir = os.path.join(src_dir, 'Cat')
src_dog_dir = os.path.join(src_dir, 'Dog')

dst_train_dir = os.path.join(dst_dir, 'train')
train_cat_dir = os.path.join(dst_train_dir, 'cat')
train_dog_dir = os.path.join(dst_train_dir, 'dog')

dst_validation_dir = os.path.join(dst_dir, 'validation')
validation_cat_dir = os.path.join(dst_validation_dir, 'cat')
validation_dog_dir = os.path.join(dst_validation_dir, 'dog')

dst_test_dir = os.path.join(dst_dir, 'test')
test_cat_dir = os.path.join(dst_test_dir, 'cat')
test_dog_dir = os.path.join(dst_test_dir, 'dog')

# Make directory using os.makedirs only when directory is not exist.
os.makedirs(train_cat_dir, exist_ok=True)
os.makedirs(train_dog_dir, exist_ok=True)
os.makedirs(validation_cat_dir, exist_ok=True)
os.makedirs(validation_dog_dir, exist_ok=True)
os.makedirs(test_cat_dir, exist_ok=True)
os.makedirs(test_dog_dir, exist_ok=True)

# Generate splitted list of filename.
sizDataSet = 12500 #2000
delim1 = int(sizDataSet*0.7)
delim2 = int(sizDataSet*0.85)
train = [str(i)+'.jpg' for i in range(0,delim1)]
validation = [str(i)+'.jpg' for i in  range(delim1,delim2)]
test = [str(i)+'.jpg' for i in  range(delim2, sizDataSet)]


def move_image(src_dir,dst_dir,filename):
    src = os.path.join(src_dir,filename)
    try:
        Image.open(src)
        dst = os.path.join(dst_dir,filename)
        shutil.move(src,dst)
    except UnidentifiedImageError:
        print("PIL.UnidentifiedImageError:",src)
    except FileNotFoundError:
        print("FileNotFoundError:",src)

for file in train:
    move_image(src_cat_dir,train_cat_dir,file)
    move_image(src_dog_dir,train_dog_dir,file)

for file in validation:
    move_image(src_cat_dir,validation_cat_dir,file)
    move_image(src_dog_dir,validation_dog_dir,file)

for file in test:
    move_image(src_cat_dir,test_cat_dir,file)
    move_image(src_dog_dir,test_dog_dir,file)

# Remove sources (It is no longer needed.)
shutil.rmtree(src_dir)
