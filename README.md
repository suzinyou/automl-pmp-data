# automl-pmp-data
Data generation code

## Requirements
* scipy
* numpy
* scikit-learn
* xgboost

To install xgboost, if you are using Linux, simply run
```bash
$ pip install xgboost
```

If you are in OS X, follow these steps:

First, install gcc. This will take about **30 minutes**.
```bash
$ brew install gcc --without-multilib
```
Once done, do this:
```bash
$ git clone --recursive https://github.com/dmlc/xgboost
$ cd xgboost
```
Also, check your gcc version by typing 
```bash
$ brew ls gcc
```
Now we'll make some changes in `make/config.mk`. Uncomment the export commands and change the gcc version to yours.
```
# xgboost/make/config.mk
export CC = /usr/local/Cellar/gcc/6.3.0_1/bin/gcc-6
export CXX = /usr/local/Cellar/gcc/6.3.0_1/bin/g++-6
```
Save the changes. Then type in your terminal
```bash
$ cp make/config.mk config.mk; make -j4
```
