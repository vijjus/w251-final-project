### Import MTCNN

From https://github.com/ipazc/mtcnn
Currently MTCNN is only supported on Python 3.4 onwards.  
It can be installed through pip:

``` $ pip install mtcnn ```

Docker container I used for HW3 has python 2.7.17 and cv2 3.2.0

My TX2 has python 3

root@kborojerdi-desktop:/data/w251/HW3# python3  
Python 3.6.9 (default, Nov  7 2019, 10:44:02)  
[GCC 8.3.0] on linux  
Type "help", "copyright", "credits" or "license" for more information.  
    >>> import cv2  
    >>> cv2.__version__  
    '4.1.1'  

root@kborojerdi-desktop:/data/w251/HW3# python3 -m pip install mtcnn
Collecting mtcnn
  Downloading https://files.pythonhosted.org/packages/67/43/abee91792797c609c1bf30f1112117f7a87a713ebaa6ec5201d5555a73ef/mtcnn-0.1.0-py3-none-any.whl (2.3MB)
    100% |████████████████████████████████| 2.3MB 304kB/s 
Collecting opencv-python>=4.1.0 (from mtcnn)
  Could not find a version that satisfies the requirement opencv-python>=4.1.0 (from mtcnn) (from versions: )
No matching distribution found for opencv-python>=4.1.0 (from mtcnn)

I also tried running this in a contianer with openCV built from source. See attached dockerfile. I got the same error.
