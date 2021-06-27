import random
import math
import os
class NeuralNetwork:
    def __init__(self,dimensions,g_weights=None):
        #dimensions input as a list of sizes eg. [1,1,1]
        self.layers= len(dimensions)
        self.dim = dimensions
        if g_weights == None:
            w = []
            for i in range(len(dimensions)):
                if i==self.layers-1:
                    continue
                nw = []
                for j in range(dimensions[i]):
                    l = []
                    for k in range(dimensions[i+1]):
                        l.append(random.random())
                    nw.append(l)
                w.append(nw)
        else:
            w = g_weights
        self.weights = w

    #Sigmoid activation function
    def sigmoid(self,gamma):
        if gamma < 0:
            return 1 - 1 / (1 + math.exp(gamma))
        else:
            return 1 / (1 + math.exp(-gamma))

    #Derivative of sigmoid activation function for backpropagation
    def sigmoid_derivative(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    #Calculates the derivitive of the cost function
    def cost_derivative(self,outputlayer,actualoutputlayer):
        ans = []
        for i in range(len(outputlayer[0])):
            ans.append(outputlayer[0][i]-actualoutputlayer[i])
        return ans

    #Computes the dotproduct of an 1 by n matrix and a n by m maxtrix
    def dotproduct_obn(self,matrix1,matrix2):
        #matrix1 is the 1 by n matrix
        #matrix2 is the n by m matrix
        ans = []
        for i in range(len(matrix2[0])):
            s = 0
            for j in range(len(matrix2)):
                s+= matrix1[j]*matrix2[j][i]
            ans.append(s)
        return ans

    #Calculates the dot product of any two matrices
    def dotproduct(self,matrix1,matrix2):
        #matrix1 is the x by n
        #matrix2 is the n by m
        if len(matrix1[0]) == len(matrix2):
            pass
        else:
            x = matrix2
            matrix2 = matrix1
            matrix1 = x
        n = len(matrix2)
        ans= []
        for row in range(len(matrix1)):
            r = []
            for column in range(len(matrix2[0])):
                c = 0
                for i in range(n):
                    c+=matrix1[row][i]*matrix2[i][column]
                r.append(c)
            ans.append(r)
        return ans

    #Performs a transpose on an array flipping the x and y axes
    def transpose(self,m):
        return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

    #Returns an output that the network computes for a given input
    def feedforward(self,inputlayer):
        curr_layer = inputlayer
        for layer_count in range(len(self.weights)):
            layer_matrix = self.dotproduct(curr_layer,self.weights[layer_count])
            curr_layer = [[]]
            for i in layer_matrix[0]:
                curr_layer[0].append(self.sigmoid(i))
        return curr_layer

    #Calculates the graident for each weight based on a training case
    def backprop(self,inputlayer, actualoutputlayer):
        weight_grad = []
        dimensions = self.dim
        for i in range(len(dimensions)):
            if i == self.layers - 1:
                continue
            weight_grad.append([[0]*(dimensions[i+1])]* dimensions[i])
        zs = []
        activations = [inputlayer]
        curr_layer = inputlayer
        for layer_count in range(len(self.weights)):
            layer_matrix = self.dotproduct(curr_layer, self.weights[layer_count])
            zs.append(layer_matrix[0])
            curr_layer = [[]]
            for i in layer_matrix[0]:
                curr_layer[0].append(self.sigmoid(i))
            activations.append(curr_layer)
        outputlayer = curr_layer
        delta = self.cost_derivative(outputlayer,actualoutputlayer)
        for i in range(len(delta)):
            delta[i]*=self.sigmoid_derivative(zs[-1][i])
        delta = [delta]
        tp_a = self.transpose(activations[-2])
        if len(delta[0]) == len(tp_a) and len(delta)==len(tp_a[0]):
            if len(delta[0])>len(delta):
                weight_grad[-1] = self.dotproduct(tp_a, delta)
            else:
                weight_grad[-1] = self.dotproduct(delta, tp_a)
        elif len(delta[0]) == len(tp_a):
            weight_grad[-1] = self.dotproduct(delta, tp_a)
        else:
            weight_grad[-1] =self.dotproduct(tp_a, delta)
        for i in range(2,self.layers):
            tp_w = self.transpose(self.weights[-i+1])
            delta = self.dotproduct(tp_w,delta)
            for j in range(len(delta)):
                delta[j][0]*=self.sigmoid_derivative(zs[-i][j])
            tp_a = self.transpose(activations[-i-1])
            if len(delta[0]) == len(tp_a):
                weight_grad[-i] = self.dotproduct(tp_a, delta)
            else:
                weight_grad[-i] = self.dotproduct(delta, tp_a)
        return weight_grad

    #Utility function for debugging
    def print_info(self):
        print(self.layers)
        print()
        print(self.dim)
        print()
        for i in self.weights:
            for j in i:
                print(j)
            print()
        print('-'*50+'\n')

    #Performs a stochastic gradient descent
    def stochasticGD(self,training_data,batchsize,lr,epoch,testdata=None):
        n = len(training_data)
        print('Base: '+ self.testself(testdata)+'\n')

        for e in range(epoch):
            random.shuffle(training_data)
            batches = []
            for i in range(0,n,batchsize):
                batches.append(training_data[i:i+batchsize])
            for batch in batches:
                self.process_batch(batch,lr)
            self.writeweights(str(epoch - e) + 'epoch')
            print('epoch {0} done'.format(epoch - e))
            if testdata:
                print(self.testself(testdata))
            print()

    #Applies backpropagation to mini-batches to train
    def process_batch(self,batch,lr):
        w = []
        dimensions = self.dim
        for i in range(len(dimensions)):
            if i == self.layers - 1:
                continue
            w.append([[0] * (dimensions[i + 1])] * dimensions[i])
        for training_case in batch:
            inputlayer = training_case[0]
            outputlayer = training_case[1]
            w_grad = self.backprop(inputlayer,outputlayer)
            for layer in range(len(w)):
                for i in range(len(w[layer])):
                    for j in range(len(w[layer][i])):
                        w[layer][i][j]+= w_grad[layer][i][j]
        for layer in range(len(w)):
            for i in range(len(w[layer])):
                for j in range(len(w[layer][i])):
                    self.weights[layer][i][j]-=w[layer][i][j]


    def writeweights(self,filename):
        fout = open(filename,'w')
        w = self.weights
        for layer in range(len(w)):
            for i in range(len(w[layer])):
                l = ''
                for j in range(len(w[layer][i])):
                    l+=str(w[layer][i][j])
                    if (j!= len(w[layer][i])-1):
                        l+=' '
                fout.write(l+'\n')
            if layer!= (len(w)-1):
                fout.write('\n')

    def readweights(self):
        fin = open('1epoch')
        w = fin.readlines()
        l = 0
        i = 0
        for line in w:
            if line == '\n':
                l+=1
                i = 0
                continue
            line = line.strip()
            line = list(map(float,line.split(' ')))
            self.weights[l][i] = line
            i+=1
        fin.close()

    def testself(self,testdata):
        print(self.weights)
        correct = 0
        total = 0
        for i in testdata:
            ans = self.feedforward(i[0])
            print(ans,i[1],ans[0].index(max(ans[0])),i[1].index(max(i[1])))
            if ans[0].index(max(ans[0])) == i[1].index(max(i[1])):
                correct+=1
            total+=1
        return 'Correct: {0}; Total: {1}; %: {2}'.format(correct,total,correct/total)
