#!/usr/bin/env python
# coding: utf-8

# In[7]:


# python notebook for Make Your Own Neural Network
# code for a 3-layer neural network, and code for learning the MNIST dataset
# this version trains using the MNIST dataset, then tests on our own images
# (c) Tariq Rashid, 2016
# license is GPLv2


# In[8]:


import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
import matplotlib.pyplot
import csv
import scipy.special
import scipy.ndimage
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# In[9]:


# helper to load data from PNG image files
import imageio


# In[10]:


# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        self.wih = numpy.load('save_wih.npy')
        self.who = numpy.load('save_who.npy')

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass


    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T #tao mang cua mang va dao mang ([a,b]->[[a],b])
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs) # nhan 2 ma tran trong so voi ma tran dau vao de dc gia tri tai cac nut lop an
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # thuc hien ham kich hoat signoid
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs) 
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# In[11]:


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)


# test with our own image 

# In[12]:


# test the neural network with our own images

from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog



root = Tk()
root.geometry("300x300")
root.iconbitmap('some images/icon.ico')
root.title("Nhan dien chu so viet tay")

#-----------------------------------------------------
def chonanh():
    global img
    global frame2
    root.filename = filedialog.askopenfilename(initialdir = "my_own_images/", title = "Chon anh", filetypes = (("png file","*.png"),("all files","*.*")))
    img = ImageTk.PhotoImage((Image.open(root.filename)).resize((250, 250), Image.ANTIALIAS))
    frame2= LabelFrame(root, text ="KET QUA",padx=5,pady=5)
    frame2.pack(padx=10, pady=10)
    imglb = Label(frame2, image = img).pack()
    
frame = LabelFrame(root ,text = "CHON ANH DE",padx=5,pady=5)
frame.pack(padx=1, pady=1)

b = Button(frame , text = "CHON ANH",command = chonanh)
b.pack()

#--------------------------------------------------------

# load image data from png files into an array
#print ("loading ... my_own_images/2.png")
#img_array = imageio.imread('my_own_images/2.png', as_gray=True)
    
# reshape from 28x28 to list of 784 values, invert values
#img_data  = 255.0 - img_array.reshape(784)
    
# then scale data to range from 0.01 to 1.0
#img_data = (img_data / 255.0 * 0.99) + 0.01

# plot image
#matplotlib.pyplot.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None')

#-----------------------------------------------------
def test():
    global img_array
    img_array = imageio.imread(root.filename, as_gray=True)
    img_data  = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    outputs = n.query(img_data)
    label = numpy.argmax(outputs)
    label1 = Label(frame2, text = "Day la so "+ str(label)).pack() 
    a["state"] = DISABLED
    b["state"] = DISABLED
    

frame1 = LabelFrame(root, text ="query",padx=5,pady=5)
frame1.pack(padx=1, pady=1)

a = Button(frame1, text = "RUN",command = test)
a.pack()


#----------------------------------------------------------
def xoa():
    frame2.pack_forget()
    a["state"] = NORMAL
    b["state"] = NORMAL
    pass
    
c = Button(frame1, text = "Xoa ket qua", command = xoa)
c.pack()


#-------------------------------------------------------
e = Entry(root)
e.pack()
def add():
    img_data = (255.0 - img_array.reshape(784)).astype(int)
    label = int(e.get())
    record = numpy.append(label,img_data)
    with open("mnist_dataset/mnist_train_notest.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(record)
buttonadd = Button(root,text= "add mnis", command = add )
buttonadd.pack()

def save(self):
    numpy.save('save_wih', self.wih)
    numpy.save('save_who', self.who)
    pass

def train():
    training_data_file = open("mnist_dataset/mnist_train_notest.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    epochs = 10
    for e in range(epochs):
        # split the record by the ',' commas
        all_values = training_data_list[-1].split(',') # chia chuoi thanh 1 chuoi loai bo dau ,
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # quy mo dau vao pham vi 0,01 den 1
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01 # tao ra 1 mang 0.01 (mang nay dien ta dau ra)
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99 # gan gia tri o thu all_values[0] - label cua mang dau ra la 0.99 tu la 
        n.train(inputs, targets)
        ## create rotated variations
        # rotated anticlockwise by x degrees
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        # rotated clockwise by x degrees
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)
    save(n)
    
buttonatrain = Button(root,text= "train mnis", command = train )
buttonatrain.pack()




root.mainloop()


# In[ ]:




