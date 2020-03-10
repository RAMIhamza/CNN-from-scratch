import numpy as np
class conv():
    def __init__(self,n_filters):
        self.n_filters=n_filters
        self.filters=np.random.randn(n_filters,3,3)/9
    def forward(self,input_image):
        self.input_image=input_image
        output = np.zeros((input_image.shape[0]-3+1,input_image.shape[1]-3+1,self.n_filters))
        self.retrieve=dict()
        for i in range(input_image.shape[0]-3+1):
            for j in range (input_image.shape[1]-3+1):
                retrieved_image=input_image[i:i+3,j:j+3,0]
                output[i,j]=np.sum(np.sum(retrieved_image*self.filters,axis=1),axis=1) #element wise product and sum
                self.retrieve[(i,j)]=retrieved_image
        return output
    def backward(self,last_gradients,lr=.005):
        new_last_gradients=np.zeros(self.filters.shape)
        for i in range(self.input_image.shape[0]-3+1):
            for j in range(self.input_image.shape[1]-3+1):    
                for f in range (self.n_filters):
                    new_last_gradients[f]+=last_gradients[i,j,f]*self.retrieve[(i,j)]
        self.filters -= lr * new_last_gradients
        return new_last_gradients


class maxpooling():
    def __init__(self,pool_size):
        self.pool_size=pool_size
        self.indices=None
    def forward(self,input_image):
        self.input_image=input_image
        self.binary_matrix=np.zeros(input_image.shape)
        d1,d2,n_filters=input_image.shape
        output=np.zeros((d1//2,d2//2,n_filters))
        indices=[]
        for i in range(d1//2):
            for j in range(d2//2):
                    output[i,j]=input_image[i*2:(i+1)*2,j*2:(j+1)*2].max(axis=(0,1))
                    indices=input_image[i*2:(i+1)*2,j*2:(j+1)*2].argmax(axis=0)  #or axis=1
                    #construction of a binary matrix
                    for f in range (n_filters):
                        i_=indices[0][f]
                        j_=indices[1][f]
                        self.binary_matrix[i*2+i_,j*2+j_,f]=1
        return output
    def backward(self,last_gradients):
        d1,d2,n_filters=self.input_image.shape
        new_last_gradients=np.zeros((2,2))
        for i,arr in enumerate(np.ndarray.flatten(last_gradients)):
            new_last_gradients=np.vstack((new_last_gradients,np.full((2,2),arr)))
        new_last_gradients=new_last_gradients[2:].reshape(last_gradients.shape[0]*2,last_gradients.shape[1]*2,last_gradients.shape[2])
        new_last_gradients=new_last_gradients * self.binary_matrix
        return new_last_gradients
  
class fully_connected():
    def __init__(self,input_dimension,output_neurons):
        self.input_dimension=input_dimension
        self.output_neurons=output_neurons
        self.weights=np.random.randn(input_dimension,output_neurons)/input_dimension
        self.biases=np.random.randn(output_neurons)
    def flatten(self,input):
        return input.flatten()
    def forward(self,input_image):
        self.last_shape=input_image.shape
        input_image = self.flatten(input_image)
        self.previous_input=input_image
        output=np.dot(input_image,self.weights)+self.biases
        output=np.exp(output)
        self.output_exp=output
        self.output_s=output/np.sum(output,axis=0)
        self.sum=np.sum(output,axis=0)
        return output/np.sum(output,axis=0)
    def backward(self,input_image,label,conv_,pool,lr=0.005):
        output = conv_.forward(input_image)
        output = pool.forward(output)
        output=self.forward(output)
        grads=np.zeros(10)
        grads[label]=-1/output[label]
        for i in range(len(grads)):
            if grads[i]==0:
                continue
            else:
                # Grads output softmax/input softmax
                d_outs_inps=-self.output_s[i]*self.output_s
                d_outs_inps[i]=self.output_exp[i]*(self.sum-self.output_exp[i])/(self.sum)**2 #Compute gradients for softmax input and output
                #grads input softmax/Weights;biases,previous input(after flatten)
                d_inps_weights=self.previous_input
                d_inps_b=1
                d_inps_input=self.weights
                #Final gradients
                d_L_inps=grads[i]*d_outs_inps
                
                d_L_weights=d_inps_weights[np.newaxis].T @ d_L_inps[np.newaxis]
                d_L_biases=d_L_inps * d_inps_b
                d_L_inputs=d_inps_input @ d_L_inps
                
            self.weights -= lr * d_L_weights
            self.biases -= lr * d_L_biases
                
        return d_L_inputs.reshape(self.last_shape)
        





































