DigitClassifier
===============

Code for kaggle.com digit classification competition

## Usage:
* Run on a sub-sample, add metadata to the image with the number of pixels above 0 per row, and per column:
  python -i main.py --train_csv train.csv  --plots --debug --n_train 2000 --n_test 2000 --scale scale --gamma 0.002 --C 5 --counts 
* Rotate the images so that the principal axis lies along the vertical:
  python -i main.py --train_csv train.csv  --plots --debug --n_train 2000 --n_test 2000 --scale scale --gamma 0.002 --C 5  --counts --rotate
* Use PCA decomposition, keeping the components that describe 99.9% of the variance
  python -i main.py --train_csv train.csv  --plots --debug --n_train 2000 --n_test 2000 --scale scale --gamma 0.002 --C 5  --counts --rotate --pca 0.99
* Apply edge detection (and replace the image)
  python -i main.py --train_csv train.csv  --plots --debug --n_train 2000 --n_test 2000 --scale scale --gamma 0.002 --C 5  --counts --rotate --pca 0.99 --edges edges
* Classify the testing dataset and write out the results to final.csv:
  python -i main.py --train_csv train.csv  --plots --debug --n_train 42000 --n_test 0 --scale scale --gamma 0.002 --C 5  --counts --rotate --pca 0.99 --edges edges --final_csv test.csv

## Inputs:
Input csv files available from: https://www.kaggle.com/c/digit-recognizer/data

Alternative input format using the MNIST files from Yann LeCunn's site: http://yann.lecun.com/exdb/mnist/

## External dependencies:

    matplotlib
      pyplot
    sklearn
      PCA
      RandomForestClassifier
      grid_search
      LDA
      metrics
      preprocessing
      svm