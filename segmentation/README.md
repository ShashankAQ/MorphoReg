# Segmentation of PBMCs's usingDeepLabv3Plus

Training DeepLabv3 on a custom dataset of stained and unstained PBMC's

## Dataset Preview
The dataset consists of grayscale .tif images of PBMC cells obtained through flow cytometry, categorized into stained and unstained samples of 30,000 samples each. It includes various types of granulocytes and monocytes, along with exceptions such as debris and possibly lymphocytes. 

![2](https://github.com/user-attachments/assets/3fa59927-41fa-4829-a698-ab4d12659761)

## Making Predictions with the Trained Model

### 1. Download the pre-trained Model

Download the custom-trained DeepLabv3+ ResNet-101 backbone from this link : [google drive](https://drive.google.com/file/d/1PLw7U5-MrrFQJ4oZue5TvnBdQ2x6IQ3a/view?usp=sharing)

### 2. Prediction

```python
git clone https://github.com/ShashankAQ/Segmentation-of-PBMC-s-using-DeepLabv3Plus.git

```
change directory

```python
cd Segmentation-of-PBMC-s-using-DeepLabv3Plus

```
Single image: 

```python
python predict.py --mode single --image_path path/to/image.jpg --ckpt path/to/model.pth --output_image result.png
```
Image folder :
```python
python predict.py --mode batch --input_dir path/to/images --output_dir path/to/save/results --ckpt path/to/model.pth
```
Predict using app:
```python
pip install streamlit
```
Make sure you replace the model directory in the app.py
```python
streamlit run app.py
```

![image](https://github.com/user-attachments/assets/b2476eda-1760-4b95-afea-985f0bd31df1)

## Training the model
Choose the required architecture,the dataset was trained on using the deeplabv3plus_resnet101 architecture
### 1. Available Architectures
| DeepLabV3    |  DeepLabV3+        |
| :---: | :---:     |
|deeplabv3_resnet50|deeplabv3plus_resnet50|
|deeplabv3_resnet101|deeplabv3plus_resnet101|
|deeplabv3_mobilenet|deeplabv3plus_mobilenet ||
|deeplabv3_hrnetv2_48 | deeplabv3plus_hrnetv2_48 |
|deeplabv3_hrnetv2_32 | deeplabv3plus_hrnetv2_32 |
|deeplabv3_xception | deeplabv3plus_xception |

### 2. Performance on Pascal VOC2012 Aug (21 classes, 513 x 513)
|  Model          | Batch Size  | FLOPs  | train/val OS   |  mIoU        | Dropbox  | Tencent Weiyun  | 
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: | :----:   |
| DeepLabV3-MobileNet       | 16      |  6.0G      |   16/16  |  0.701     |    [Download](https://www.dropbox.com/s/uhksxwfcim3nkpo/best_deeplabv3_mobilenet_voc_os16.pth?dl=0)       | [Download](https://share.weiyun.com/A4ubD1DD) |
| DeepLabV3-ResNet50         | 16      |  51.4G     |  16/16   |  0.769     |    [Download](https://www.dropbox.com/s/3eag5ojccwiexkq/best_deeplabv3_resnet50_voc_os16.pth?dl=0) | [Download](https://share.weiyun.com/33eLjnVL) |
| DeepLabV3-ResNet101         | 16      |  72.1G     |  16/16   |  0.773     |    [Download](https://www.dropbox.com/s/vtenndnsrnh4068/best_deeplabv3_resnet101_voc_os16.pth?dl=0)       | [Download](https://share.weiyun.com/iCkzATAw)  |
| DeepLabV3Plus-MobileNet   | 16      |  17.0G      |  16/16   |  0.711    |    [Download](https://www.dropbox.com/s/0idrhwz6opaj7q4/best_deeplabv3plus_mobilenet_voc_os16.pth?dl=0)   | [Download](https://share.weiyun.com/djX6MDwM) |
| DeepLabV3Plus-ResNet50    | 16      |   62.7G     |  16/16   |  0.772     |    [Download](https://www.dropbox.com/s/dgxyd3jkyz24voa/best_deeplabv3plus_resnet50_voc_os16.pth?dl=0)   | [Download](https://share.weiyun.com/uTM4i2jG) |
| DeepLabV3Plus-ResNet101     | 16      |  83.4G     |  16/16   |  0.783     |    [Download](https://www.dropbox.com/s/bm3hxe7wmakaqc5/best_deeplabv3plus_resnet101_voc_os16.pth?dl=0)   | [Download](https://share.weiyun.com/UNPZr3dk) |

### 3. Performance on Cityscapes (19 classes, 1024 x 2048)

|  Model          | Batch Size  | FLOPs  | train/val OS   |  mIoU        | Dropbox  |  Tencent Weiyun  |
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: |  :----:   |
| DeepLabV3Plus-MobileNet   | 16      |  135G      |  16/16   |  0.721  |    [Download](https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=0) | [Download](https://share.weiyun.com/aSKjdpbL) 
| DeepLabV3Plus-ResNet101   | 16      |  N/A      |  16/16   |  0.762  |    [Download](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing) | N/A |

## Steps to train

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

Dataset format:
Make sure the dataset format is exactly like this 
```
/datasets
    /data
        /VOCdevkit 
            /VOC2012
                /ImageSets
                    Segmentation\
                            trainval.txt # <= labels of training images + labels of val images
                            train.txt # <= labels of training images (images have to be in .jpg)
                            val.txt  # <= labels of val images (masks have to be in .png)

                /SegmentationClass # <= Segmentation masks

               /JPEGImages # <= images
                ...
            ...
        ...
```
### 3. Training on Pascal VOC2012 Aug(21 classes)

#### 3.1 Visualize training (Optional)
Start visdom sever for visualization. Please remove '--enable_vis' if visualization is not needed. 

```bash
# Run visdom server on port 28333
visdom -port 28333
```
#### 3.2 Training with OS=16
Run main.py with *"--year 2012_aug"* to train your model on Pascal VOC2012 Aug. You can also parallel your training on 4 GPUs with '--gpu_id 0,1,2,3'

**Note: There is no SyncBN in this repo, so training with *multple GPUs and small batch size* may degrades the performance. See [PyTorch-Encoding](https://hangzhang.org/PyTorch-Encoding/tutorials/syncbn.html) for more details about SyncBN**

The parameters below were chosen based on the dataset size and available computational resources. For further customization, refer to the main.py file to adjust or initialize additional training parameters as needed.

```bash
!python main.py \
  --model deeplabv3plus_resnet101 \
  --gpu_id 0 \
  --year 2012 \
  --crop_val \
  --lr 0.0005 \
  --crop_size 513 \
  --batch_size 8 \
  --output_stride 16 \
  --data_root "/content/drive/MyDrive/VOC_Dataset" \
  --ckpt "/content/drive/MyDrive/model.pth" \
  --loss_type focal_loss \
  --lr_policy poly \
  --total_itrs 2500 \
  --save_val_results \
  --random_seed 42
```
The main.py code was adjusted from SGD to AdamW for more stable convergence and better generalization.
Poly learning rate scheduling (PolyLR) was paired with AdamW as it enables smooth and gradual learning rate decay, which works well for small datasets and helps maintain performance over longer training iterations in semantic segmentation tasks.

#### 3.3 Continue training

Run main.py with '--continue_training' to restore the state_dict of optimizer and scheduler from YOUR_CKPT.

```bash
python main.py ... --ckpt YOUR_CKPT --continue_training
```

#### 3.4. Testing

Results will be saved at ./results.

```bash
!python main.py --model deeplabv3plus_resnet101 \
    --gpu_id 0 \
    --year 2012 \
    --crop_val \
    --lr 0.01 \
    --crop_size 513 \
    --batch_size 4 \
    --output_stride 16 \
    --data_root "/content/drive/MyDrive/VOC_Dataset" \
    --ckpt "/content/DeepLabV3Plus-Pytorch/checkpoints/latest_deeplabv3plus_resnet101_voc_os16.pth" \
    --test_only \
    --save_val_results

```
 For additional configuration details and advanced settings, refer to the original VainF/DeepLabV3Plus-Pytorch [repository](https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master)






















