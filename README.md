# SEDCNN-for-SAXS-and-WAXD
-----
In this warehouse is the code of the article 'A SEDCNN Machine Learning Model for Textured SAXS/WAXD Image Denoising'
-----
## Prerequisites
1. Code is intended to work with 
2. ```Python 3.8.x```(it hasn't been tested with previous versions)
3. ```pytorch 1.9.1```
4. ```numpy 1.21.2```
5. ```skimage 1.7.1```
6. ```scipy 0.18.3```
7. ```sklearn 0.18.2```
8. ```pyFAI 0.20.0```
9. ```xlwt 1.3.0```
10. ```seaborn 0.11.2```
11. ```argparse```
-----
The network framework and specific content of SEDCNN4 are as follows
![image](https://github.com/zzZhou8/SEDCNN-for-SAXS-and-WAXD/blob/main/img/networks.png)
-----
Its optimization results on SAXS images are shown below(the bottom three figures show the radial integration) 
![image](https://github.com/zzZhou8/SEDCNN-for-SAXS-and-WAXD/blob/main/img/SAXS.png)
-----
The optimization results on WAXD images are shown below
![image](https://github.com/zzZhou8/SEDCNN-for-SAXS-and-WAXD/blob/main/img/WAXD.png)
-----
## DATASET
1. All of our input data is in '.npy' format.
2. All of our output data is in '.tif' format.
-----
## Use
1. run `python main.py --mode='tarin'` to training.
2. run `python main.py --mode='test' --test_epochs=500` to test.
3. run `python main.py --Loop_test='True'` Otherwise, the network will import each saved model and see how it performs on the test set (no output image will be generated, but numerical fitting and comparison will still be performed). 
4. run `python main.py --Loop_test='False'` When loop testing is not selected , the model determined by test_epochs will be directly imported and the image prediction and numerical comparison will be performed.
-----
## Explain
In the SEDCNN_WAXD test, if you want to use the Revised_33 image used for the article (we have saved it to Revised_WAXD_tif), please use the model saved for 1000 epochs.  (Of course, it is basically the same as the test result of the model saved in 500 rounds, so it is hereby noted.) 
