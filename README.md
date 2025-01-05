# 1st-DL-CVMarathon

## 一、專題摘要

期末專題主題：浣熊與袋鼠辨識模型
期末專題基本目標 
以yolov3訓練模型為基礎，訓練一個能夠辨識浣熊(raccoon)與袋鼠(kangaroo)的模型。
以訓練好的模型辨識浣熊與袋鼠的測試圖片及影片。

## 三、成果展示

### 1.Loss指標
![Loss](https://clubfile.cupoy.com/00000170BAC9DD980000001A6375706F795F72656C656173654B5741535354434C55424E455753/1574048361442/large)

| Epoch | 1   | 10  | 20  | 30  | 40  | 50  |
|-------|-----|-----|-----|-----|-----|-----|
| Loss  | 1761 | 26  | 20  | 18  | 18  | 17  |

Loss下降的很快，但是在20 Epoch後就下降緩慢



### 2.不同batch_size的辨識效果比較

以batch_size=4與batch_size=8做比較

![](http://clubfile.cupoy.com/00000170BAC9DD980000001A6375706F795F72656C656173654B5741535354434C55424E455753/1574048361446/large)

![](http://clubfile.cupoy.com/00000170BAC9DD980000001A6375706F795F72656C656173654B5741535354434C55424E455753/1574048361445/large)

![](http://clubfile.cupoy.com/00000170BAC9DD980000001A6375706F795F72656C656173654B5741535354434C55424E455753/1574048361447/large)

![](http://clubfile.cupoy.com/00000170BAC9DD980000001A6375706F795F72656C656173654B5741535354434C55424E455753/1574048361448/large)

batch_size增加雖然會使用更多的記憶體資源，但確實可以增加辨識度。



### 3.影片辨識結果

[kangaroo.mp4](https://drive.google.com/open?id=1-Fc7peSuGSGoAWvGsDwKdq11cB3kS5tS)
![](http://clubfile.cupoy.com/00000170BAC9DD980000001A6375706F795F72656C656173654B5741535354434C55424E455753/1574048361468/large)

[raccoon.mp4](https://drive.google.com/open?id=1-Ilnwvv_V2lnJCYQOSwhuWQoEtmYhrx1)
![](http://clubfile.cupoy.com/00000170BAC9DD980000001A6375706F795F72656C656173654B5741535354434C55424E455753/1574048361477/large)

但正面圖片的辨識率不錯。


## 二、實作方法介紹

1.使用的程式碼介紹

1.1前置準備(安裝keras，確認tensorflow版本，將google drive連上colob，下載yolov3及權重，下載浣熊及袋鼠訓練集)
```
# 確保 colob 中使用的 tensorflow 是 1.x 版本而不是 tensorflow 2
%tensorflow_version 1.x 
import tensorflow as tf
print(tf.__version__)

# 需要安裝 keras 2.2.4 的版本, 否則訓練時會出現 error
!pip install keras==2.2.4
import keras
print(keras.__version__)

# 將 google drive 掛載在 colob
from google.colab import drive
drive.mount('/content/gdrive')
%cd '/content/gdrive/My Drive'

# 下載基於 keras 的 yolov3 程式碼
import os
if not os.path.exists("keras-yolo3") :
  !git clone https://github.com/qqwweee/keras-yolo3
else :
  print("keras-yolo3 exists")
%cd keras-yolo3

# model_data/yolo.h5 模型 & 權重
# 下載 yolov3 的網路權重，並且把權重轉換為 keras 能夠讀取的格式
if not os.path.exists("model_data/yolo.h5"):
  print("Model doesn't exist, downloading...")
  os.system("wget https://pjreddie.com/media/files/yolov3.weights")
  print("Converting yolov3.weights to yolo.h5...")
  os.system("python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5")
else:
  print("Model exist")

# 下載 raccoon 與 kangaroo 的資料集
if not os.path.exists("raccoon_dataset"):
  !git clone https://github.com/experiencor/raccoon_dataset.git  # 下載 raccoon_dataset 資料集
else:
  print("raccoon_dataset exists")

if not os.path.exists("kangaroo"):
  !git clone https://github.com/experiencor/kangaroo.git  # 下載 kangaroo 資料集
else:
  print("kangaroo exists")

# 下載浣熊與袋鼠影片
if not os.path.exists("video"):
  !mkdir video
  !wget -c 'https://cvdl.cupoy.com/HomeworkAction.do?op=getHomeworkFileContent&hwid=D49&filepath=Raccoon.mp4' -O 'video/Raccoon.mp4'      # Raccoon 測試影片
  !wget -c 'https://cvdl.cupoy.com/HomeworkAction.do?op=getHomeworkFileContent&hwid=D49&filepath=Kangaroo.mp4' -O 'video/Kangaroo.mp4'    # kangaroo 測試影片
else:
  print("video exists")
# 確保 colob 中使用的 tensorflow 是 1.x 版本而不是 tensorflow 2
%tensorflow_version 1.x 
import tensorflow as tf
print(tf.__version__)

# 需要安裝 keras 2.2.4 的版本, 否則訓練時會出現 error
!pip install keras==2.2.4
import keras
print(keras.__version__)

# 將 google drive 掛載在 colob
from google.colab import drive
drive.mount('/content/gdrive')
%cd '/content/gdrive/My Drive'

# 下載基於 keras 的 yolov3 程式碼
import os
if not os.path.exists("keras-yolo3") :
  !git clone https://github.com/qqwweee/keras-yolo3
else :
  print("keras-yolo3 exists")
%cd keras-yolo3

# model_data/yolo.h5 模型 & 權重
# 下載 yolov3 的網路權重，並且把權重轉換為 keras 能夠讀取的格式
if not os.path.exists("model_data/yolo.h5"):
  print("Model doesn't exist, downloading...")
  os.system("wget https://pjreddie.com/media/files/yolov3.weights")
  print("Converting yolov3.weights to yolo.h5...")
  os.system("python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5")
else:
  print("Model exist")

# 下載 raccoon 與 kangaroo 的資料集
if not os.path.exists("raccoon_dataset"):
  !git clone https://github.com/experiencor/raccoon_dataset.git  # 下載 raccoon_dataset 資料集
else:
  print("raccoon_dataset exists")

if not os.path.exists("kangaroo"):
  !git clone https://github.com/experiencor/kangaroo.git  # 下載 kangaroo 資料集
else:
  print("kangaroo exists")

# 下載浣熊與袋鼠影片
if not os.path.exists("video"):
  !mkdir video
  !wget -c 'https://cvdl.cupoy.com/HomeworkAction.do?op=getHomeworkFileContent&hwid=D49&filepath=Raccoon.mp4' -O 'video/Raccoon.mp4'      # Raccoon 測試影片
  !wget -c 'https://cvdl.cupoy.com/HomeworkAction.do?op=getHomeworkFileContent&hwid=D49&filepath=Kangaroo.mp4' -O 'video/Kangaroo.mp4'    # kangaroo 測試影片
else:
  print("video exists")
```
1.2資料集解析 (處理 xml 資料)
```
import numpy as np
# 訓練模型時需使用的 annotation 檔名, 若已經做好轉換, 則不會每次再重新跑這段轉換的程式碼
if not os.path.exists("train_labels.txt"):
  import xml.etree.ElementTree as ET # 載入能夠 Parser xml 文件的 library
  
  sets=['train', 'val']

  # "raccoon", "kangaroo" 的資料類別
  classes = ["raccoon", "kangaroo"]

  # 把 annotation(.xml) 轉換到訓練時需要的資料形態
  def convert_annotation(image_id, list_file):
      in_file = open('annotation_xml/%s.xml'%(image_id))
      tree=ET.parse(in_file)
      root = tree.getroot()

      for obj in root.iter('object'):
          difficult = obj.find('difficult').text
          cls = obj.find('name').text
          if cls not in classes or int(difficult)==1: 
              continue
          cls_id = classes.index(cls)  # class index
          xmlbox = obj.find('bndbox')
          b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), 
                int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
          list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

  # 把 raccoon_dataset/images 與 kangaroo/images 檔案合併後, 當成訓練集 & 驗證集資料
  for root,dirs,files in os.walk('raccoon_dataset/images') :
    print('raccoon jpg 檔數量:', len(files))
  for root_2,dirs_2,files_2 in os.walk('kangaroo/images') :
    print('kangaroo jpg 檔數量:', len(files_2))
  # 把 files_2 合併在 files list 內
  files.extend(files_2)
  print('所有 jpg 檔數量:', len(files))
    
  jpg_ids = ''.join(files).strip().split('.jpg')[:-1]
  # 80% 檔案資料當成訓練集資料
  train_index = np.random.choice(jpg_ids, size=int(len(jpg_ids)*0.8), replace=False)
  val_index = np.setdiff1d(jpg_ids, train_index)

  !mkdir train val
  # 把訓練集資料檔索引, 放入 train 資料夾
  train_txt = open('train/train.txt', 'w')
  print("save train index at train/train.txt")       
  for train_id in train_index : 
      train_txt.write('%s' %(train_id))
      train_txt.write('\n')
  train_txt.close()

  # 把驗證集資料檔索引, 放入 val 資料夾
  val_txt = open('val/val.txt', 'w')
  print("save val index at val/val.txt")       
  for val_id in val_index : 
      val_txt.write('%s' %(val_id))
      val_txt.write('\n')
  val_txt.close()

  # 把annotation(.xml), 放入 annotation_xml 資料夾
  !mkdir annotation_xml
  !cp raccoon_dataset/annotations/*.xml ./annotation_xml
  !cp kangaroo/annots/*.xml ./annotation_xml
/**/
  # 把類別資料放入 class.txt
  class_txt = open('class.txt', 'w')
  print("save class at class.txt")       
  for class_id in classes : 
      class_txt.write('%s' %(class_id))
      class_txt.write('\n')
  class_txt.close()

  for image_set in sets:
      image_ids = open('%s/%s.txt'%(image_set, image_set)).read().strip().split()
      
      annotation_path = '%s_labels.txt'%(image_set)
      list_file = open(annotation_path, 'w')
      print("save annotation at %s" % annotation_path)
      # 處理訓練集 & 驗證集資料檔
      for image_id in image_ids:
        if 'raccoon' in image_id :
          list_file.write('./raccoon_dataset/images/%s.jpg' %(image_id))
        else :
          list_file.write('./kangaroo/images/%s.jpg' %(image_id))  
        convert_annotation(image_id, list_file)
        list_file.write('\n')
      list_file.close()
import numpy as np
# 訓練模型時需使用的 annotation 檔名, 若已經做好轉換, 則不會每次再重新跑這段轉換的程式碼
if not os.path.exists("train_labels.txt"):
  import xml.etree.ElementTree as ET # 載入能夠 Parser xml 文件的 library
  
  sets=['train', 'val']

  # "raccoon", "kangaroo" 的資料類別
  classes = ["raccoon", "kangaroo"]

  # 把 annotation(.xml) 轉換到訓練時需要的資料形態
  def convert_annotation(image_id, list_file):
      in_file = open('annotation_xml/%s.xml'%(image_id))
      tree=ET.parse(in_file)
      root = tree.getroot()

      for obj in root.iter('object'):
          difficult = obj.find('difficult').text
          cls = obj.find('name').text
          if cls not in classes or int(difficult)==1: 
              continue
          cls_id = classes.index(cls)  # class index
          xmlbox = obj.find('bndbox')
          b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), 
                int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
          list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

  # 把 raccoon_dataset/images 與 kangaroo/images 檔案合併後, 當成訓練集 & 驗證集資料
  for root,dirs,files in os.walk('raccoon_dataset/images') :
    print('raccoon jpg 檔數量:', len(files))
  for root_2,dirs_2,files_2 in os.walk('kangaroo/images') :
    print('kangaroo jpg 檔數量:', len(files_2))
  # 把 files_2 合併在 files list 內
  files.extend(files_2)
  print('所有 jpg 檔數量:', len(files))
    
  jpg_ids = ''.join(files).strip().split('.jpg')[:-1]
  # 80% 檔案資料當成訓練集資料
  train_index = np.random.choice(jpg_ids, size=int(len(jpg_ids)*0.8), replace=False)
  val_index = np.setdiff1d(jpg_ids, train_index)

  !mkdir train val
  # 把訓練集資料檔索引, 放入 train 資料夾
  train_txt = open('train/train.txt', 'w')
  print("save train index at train/train.txt")       
  for train_id in train_index : 
      train_txt.write('%s' %(train_id))
      train_txt.write('\n')
  train_txt.close()

  # 把驗證集資料檔索引, 放入 val 資料夾
  val_txt = open('val/val.txt', 'w')
  print("save val index at val/val.txt")       
  for val_id in val_index : 
      val_txt.write('%s' %(val_id))
      val_txt.write('\n')
  val_txt.close()

  # 把annotation(.xml), 放入 annotation_xml 資料夾
  !mkdir annotation_xml
  !cp raccoon_dataset/annotations/*.xml ./annotation_xml
  !cp kangaroo/annots/*.xml ./annotation_xml
/**/
  # 把類別資料放入 class.txt
  class_txt = open('class.txt', 'w')
  print("save class at class.txt")       
  for class_id in classes : 
      class_txt.write('%s' %(class_id))
      class_txt.write('\n')
  class_txt.close()

  for image_set in sets:
      image_ids = open('%s/%s.txt'%(image_set, image_set)).read().strip().split()
      
      annotation_path = '%s_labels.txt'%(image_set)
      list_file = open(annotation_path, 'w')
      print("save annotation at %s" % annotation_path)
      # 處理訓練集 & 驗證集資料檔
      for image_id in image_ids:
        if 'raccoon' in image_id :
          list_file.write('./raccoon_dataset/images/%s.jpg' %(image_id))
        else :
          list_file.write('./kangaroo/images/%s.jpg' %(image_id))  
        convert_annotation(image_id, list_file)
        list_file.write('\n')
      list_file.close()
```
1.3分配訓練資料集及驗證資料集(8:2)及設定模型參數
```
annotation_path_train = 'train_labels.txt' # 轉換好格式的 train 標註檔案
annotation_path_val = 'val_labels.txt' # 轉換好格式的 val 標註檔案
log_dir = 'logs/000/' # 訓練好的模型儲存的路徑
classes_path = 'class.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

input_shape = (416,416) # multiple of 32, hw

is_tiny_version = len(anchors)==6 # default setting
if is_tiny_version:
    model = create_tiny_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
else:
    model = create_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=30)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

with open(annotation_path_train) as f:
    lines_train = f.readlines()
with open(annotation_path_val) as f:
    lines_val = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines_train)
np.random.shuffle(lines_val)
np.random.seed(None)
num_train = len(lines_train)  # 訓練資料(80%)
num_val = len(lines_val)      # 驗證資料(20%)
annotation_path_train = 'train_labels.txt' # 轉換好格式的 train 標註檔案
annotation_path_val = 'val_labels.txt' # 轉換好格式的 val 標註檔案
log_dir = 'logs/000/' # 訓練好的模型儲存的路徑
classes_path = 'class.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

input_shape = (416,416) # multiple of 32, hw

is_tiny_version = len(anchors)==6 # default setting
if is_tiny_version:
    model = create_tiny_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
else:
    model = create_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=30)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

with open(annotation_path_train) as f:
    lines_train = f.readlines()
with open(annotation_path_val) as f:
    lines_val = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines_train)
np.random.shuffle(lines_val)
np.random.seed(None)
num_train = len(lines_train)  # 訓練資料(80%)
num_val = len(lines_val)      # 驗證資料(20%)
```
1.4載入預訓練模型
```
# convert.py '-w' : 代表只轉換權重 weights 到 model_data/yolo_weights.h5
if not os.path.exists("model_data/yolo_weights.h5"):
  print("Converting pretrained YOLOv3 weights for training")
  os.system("python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5") 
else:
  print("Pretrained weights exists")
# convert.py '-w' : 代表只轉換權重 weights 到 model_data/yolo_weights.h5
if not os.path.exists("model_data/yolo_weights.h5"):
  print("Converting pretrained YOLOv3 weights for training")
  os.system("python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5") 
else:
  print("Pretrained weights exists")
```
1.5第一階段訓練
```
# 一開始先 freeze YOLO 除了 output layer 以外的 darknet53 backbone 來 train
if True:
    model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    batch_size = 4
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model_1= model.fit_generator(data_generator_wrapper(lines_train, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines_val, batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging, checkpoint, reduce_lr])
    model.save_weights(log_dir + 'trained_weights_stage_1.h5')
# 一開始先 freeze YOLO 除了 output layer 以外的 darknet53 backbone 來 train
if True:
    model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    batch_size = 4
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model_1= model.fit_generator(data_generator_wrapper(lines_train, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines_val, batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging, checkpoint, reduce_lr])
    model.save_weights(log_dir + 'trained_weights_stage_1.h5')
```
1.6第二階段訓練
```
# Unfreeze and continue training, to fine-tune.
if True:
    # 把所有 layer 都改為 trainable
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    print('Unfreeze all of the layers.')
    # note that more GPU memory is required after unfreezing the body
    batch_size = 4 
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    hist_model= model.fit_generator(data_generator_wrapper(lines_train, batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines_val, batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=100,
        initial_epoch=50,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    
    model.save_weights(log_dir + 'trained_weights_final.h5')
# Unfreeze and continue training, to fine-tune.
if True:
    # 把所有 layer 都改為 trainable
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    print('Unfreeze all of the layers.')
    # note that more GPU memory is required after unfreezing the body
    batch_size = 4 
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    hist_model= model.fit_generator(data_generator_wrapper(lines_train, batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines_val, batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=100,
        initial_epoch=50,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    
    model.save_weights(log_dir + 'trained_weights_final.h5')
2.使用的模組介紹

# 將 train.py 所需要的套件載入
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from train import get_classes, get_anchors, create_model, create_tiny_model, data_generator, data_generator_wrapper
# 因訓練時發生 error, 故加入此程式碼 :
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 將 train.py 所需要的套件載入
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from train import get_classes, get_anchors, create_model, create_tiny_model, data_generator, data_generator_wrapper
# 因訓練時發生 error, 故加入此程式碼 :
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

## 四、結論
將batch_size從4改成16可以增加辨識率。遇到同一張圖片會同時辨識成浣熊與袋鼠的情形。袋鼠的背影無法辨識。會把狗誤認成袋鼠。
未來可以利用data augmentation的方式增加訓練資料。可以再進一步利用IoU及mAP指標檢視模型訓練成果。
