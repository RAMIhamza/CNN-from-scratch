# CNN-from-scratch

CNN using only numpy library

CNN trains on images from MNIST database stored in the kaggle competition https://www.kaggle.com/c/digit-recognizer/overview

Please put your images under model/train.csv & model/test.csv (data can be downloaded from here https://www.kaggle.com/c/digit-recognizer/data )

To train your model :

```
python main.py --n_steps 15000
```

You can modify n_steps in the arguments (Also see main.py for additional arguments)

After finishing the training plots will be saved and also all the layers' weights in ./model

Submission score on kaggle:

- n_steps = 10000    score = 0.90271

- n_steps = 15000    score = 0.89871

- n_steps = 20000    score = 0.89285



### Architecture:
conv(8,3,3), maxpooling(2), fully connected(10)
