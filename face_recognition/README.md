# Face Recognition #

Experimented with:

https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

## Steps ##

* Install cmake

```
sudo apt-get install cmake
```

* Get the dlib code

```
git clone https://github.com/davisking/dlib.git
```

* Compile dlib

```
cd dlib/
mkdir build; cd build; cmake ..; cmake --build .
```

* Install pip3

```
sudo apt install python3-pip
```

* Install other Python packages (from the dlib directory)

```
python3 setup.py install
```

Now we are able to import dlib:

```
root@final-project:/data/face/dlib# python3
Python 3.6.9 (default, Nov  7 2019, 10:44:02) 
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import dlib
>>> exit()
```

* Install Jupyter, face_recognition, matplotlib

```
pip3 install jupyter face_recognition matplotlib
```

* Run Jupyter and follow notebook

```
jupyter notebook --ip=0.0.0.0 --allow-root
```
