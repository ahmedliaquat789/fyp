# CartoonBANK
We are committed to building an cartoon community using deep generative models.

## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN
- Python 3

### Installation
- Clone the repository:
``` 
git clone https://github.com/MingtaoGuo/CartoonBANK.git
cd CartoonBANK
```
### Dependencies:  
- scikit-image=0.18.3
- torch=1.12.1
- opencv-python=4.5.3.56
- opencv-python-headless=4.5.3.56
- numpy=1.20.3

### Inference
- Download the face detection model [「yolov5s-face.onnx」](https://drive.google.com/file/d/1gcXDs4BBIOTc7RrlGrnSrJHn0wcuW_0z/view?usp=share_link), put the **yolov5s-face.onnx** into the folder **saved_models**
``` 
python inference.py --img_path examples/zjl.png --cartoon_model saved_models/style1.pth  --save_path result.jpg
```
## Cartoon BANK 
- **Cartoon Style 1** [「Pretrained model」](https://drive.google.com/file/d/1IaeyroN4rSwsSZ5eUiHyooAJXZoEI-LQ/view?usp=sharing)

|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/zjl_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/zjl_style1.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/xlb_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/xlb_style1.jpg)|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/sm_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/sm_style1.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/ngls_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/ngls_style1.jpg)|
|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/lyf_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/lyf_style1.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/mm_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/mm_style1.jpg)|

- **Cartoon Style 2** [「Pretrained model」](https://drive.google.com/file/d/1YRc1cjSYWmtdgWHXBqPKbweYcEwe_1lC/view?usp=sharing)

|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/zjl_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/zjl_style2.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/xlb_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/xlb_style2.jpg)|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/sm_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/sm_style2.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/ngls_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/ngls_style2.jpg)|
|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/lyf_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/lyf_style2.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/mm_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/mm_style2.jpg)|

- **Cartoon Style 3** [「Pretrained model」](https://drive.google.com/file/d/1XfTvYUniWoH2oS6tm-znLRoZ3QbkkPVo/view?usp=sharing)

|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/zjl_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/zjl_style3.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/xlb_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/xlb_style3.jpg)|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/sm_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/sm_style3.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/ngls_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/ngls_style3.jpg)|
|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/lyf_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/lyf_style3.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/mm_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/mm_style3.jpg)|

- **Cartoon Style 4 'Mona Lisa'** [「Pretrained model」](https://drive.google.com/file/d/1IePYGr3hnUa6OaFZV5JZUfwdCGe3lcqU/view?usp=share_link)

|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/zjl_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/zjl_style4.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/xlb_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/xlb_style4.jpg)|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/sm_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/sm_style4.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/ngls_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/ngls_style4.jpg)|
|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/lyf_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/lyf_style4.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/mm_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/mm_style4.jpg)|

- **Cartoon Style 5** [「Pretrained model」](https://drive.google.com/file/d/18Lso6NX9j9cmTLPJjvkoFPn_lkAcWsH4/view?usp=share_link)

|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/zjl_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/zjl_style5.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/xlb_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/xlb_style5.jpg)|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/sm_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/sm_style5.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/ngls_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/ngls_style5.jpg)|
|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/lyf_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/lyf_style5.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/mm_align.jpg)|![](https://github.com/MingtaoGuo/CartoonBANK/blob/main/IMGS/mm_style5.jpg)|

## Author 
Mingtao Guo
E-mail: gmt798714378 at hotmail dot com


## To be continue. 
We will update fresh cartoon model every friday night in Chinese time, Or if you'd like a custom cartoon model, upload or send the cartoon image to us. Please enjoy it :-).