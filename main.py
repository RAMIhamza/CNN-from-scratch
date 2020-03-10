import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from layers import conv,maxpooling,fully_connected

def cross_entropy_loss(output,true_label):
    return -np.log(output[true_label])
    
def accuracy(output,true_label):
    predicted_class=np.argmax(output)
    if predicted_class==true_label:
        return 1
    else :
        return 0

def forward(image, label,conv_,pool,fully_c):

  out = conv_.forward(image)
  out = pool.forward(out)
  out = fully_c.forward(out)

  # Calculate cross-entropy loss and accuracy.
  loss = cross_entropy_loss(out,label)
#   print(out[label])
  acc = accuracy(out,label)

  return out, loss, acc

def predict(image,conv_,pool,fully_c):
    
    out = conv_.forward(image)
    out = pool.forward(out)
    out = fully_c.forward(out)
    return np.argmax(out) 
  
def train_(im, label,conv_,pool,fully_c):

  # Forward
  out, loss, acc = forward(im, label,conv_,pool,fully_c)
  # Backprop
  gradient = fully_c.backward(im,label,conv_,pool)
  gradient_2 = pool.backward(gradient)
  gradient_3 = conv_.backward(gradient_2)

  return loss, acc 

print('CNN started!')
if __name__ == '__main__': 
    parser = argparse.ArgumentParser("Training settings")
    parser.add_argument("--n_steps", help="number of training steps",default=10000 ,type=int)
    parser.add_argument("--input_train_data", help="A file to parse",default="./data/train.csv", type=str)
    parser.add_argument("--input_test_data", help="A file to parse",default="./data/test.csv", type=str)
    parser.add_argument("--plot", help="True if you want to shaw and save the plots",default=True, type=bool)
    parser.add_argument("--test_mode", help="test on 2500 images",default=True, type=bool)

    
    args = parser.parse_args()
    # Load the data
    print('start loading the data')
    train = pd.read_csv(args.input_train_data)
    test = pd.read_csv(args.input_test_data)
    Y_train = train["label"][:args.n_steps]
    Y_dev = train["label"][args.n_steps:args.n_steps+2500]
    # Drop 'label' column
    X_train = train.drop(labels = ["label"],axis = 1) [:args.n_steps]
    X_dev = train.drop(labels = ["label"],axis = 1) [args.n_steps:args.n_steps+2500]
    #normalization
    # Normalize the data
    print("normalize data")
    X_train = X_train / 255.0
    print(len(X_train))
    X_dev = X_dev / 255.0

    test = test / 255.0 
    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    X_train = X_train.values.reshape(-1,28,28,1)
    X_dev = X_dev.values.reshape(-1,28,28,1)
    X_train.shape

    test = test.values.reshape(-1,28,28,1)
    # initialize weights
    conv_ = conv(8)
    pool = maxpooling(2)
    fully_c=fully_connected(13*13*8,10)
    # Train!
    loss = 0
    n_epochs=2
    losses=[]
    accs=[]
    num_correct = 0
    print('start training')
    for i, (im, label) in enumerate(zip(X_train, Y_train)):
      # print(im.shape)
    #   print(label)
      if i % 100 == 99:
        print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))
        losses.append(loss / 100)
        accs.append(num_correct)
        loss = 0
        num_correct = 0
      # print(conv_)
      l, acc = train_(im, label, conv_, pool,fully_c)
      loss += l
      num_correct += acc
    #saving the model
    np.save("./model/cnv_weights.npy",conv_.filters)
    np.save("./model/fully_c_weights.npy",fully_c.weights)
    np.save("./model/fully_c_biases.npy",fully_c.biases)
    if args.plot:
      # print(accs)
      plt.figure(figsize=(10,10))
      plt.plot(np.arange(0,len(accs)*100,100),accs)
      plt.xlabel('steps')
      plt.ylabel('accuracy')
      # plt.show()
      plt.savefig("training_accuracy.png")

      plt.figure(figsize=(10,10))
      plt.plot(np.arange(0,len(losses)*100,100),losses)
      plt.xlabel('steps')
      plt.ylabel('loss')
      # plt.show()
      plt.savefig("training_loss.png")
    if args.test_mode:
      print('start testing on 2500 images')
      loss = 0
      num_correct = 0
      for im, label in zip(X_dev, Y_dev):
        _, l, acc = forward(im, label,conv_,pool,fully_c)
        loss += l
        num_correct += acc
      # plt.plot(np.arange(0,100),np.ones(100)*20,label='test')
      num_tests = len(X_dev)
      print('Test Loss:', loss / num_tests)
      print('Test Accuracy: {} %'.format( num_correct / num_tests * 100))


      
 
