# ocr area detect by psenet
paper2018: [Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/abs/1806.02559)

* #### prepare model</br>
download the pb model https://drive.google.com/open?id=1620RtIaUbPI9OcalsUU6BhthB5pxr_CN</br>
or train by self, the repos [tensorflow_PSENet](https://github.com/liuheng92/tensorflow_PSENet)

* #### evaluate</br>
`python eval.py`</br>
it will detect the images under folder `images`, and generate the results in `results`

![](https://github.com/taylorlu/ocrDetect/blob/master/results/417505165220284184.jpg)
