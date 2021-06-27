import nnsrc,random
inputsize = 5
x = nnsrc.NeuralNetwork([inputsize,4,2])
x.print_info()
#print(x.backprop([[1]],[0.9,0]))
training_base = random.choices(range(1,2**inputsize),k=1000)
trainingdata = []
for i in training_base:
    b = bin(i)
    b = b.replace('0b','')
    b = list(map(int,list(b)))
    for j in range(inputsize-len(b)):
        b.insert(0,0)
    for j in range(len(b)):
        if b[j]==0:
            b[j] = 1
        else:
            b[j] = 100
    if i>16:
        trainingdata.append([[b], [0, 1.0]])
    else:
        trainingdata.append([[b], [1.0, 0]])
test_base = random.choices(range(1,2**inputsize),k=10)
#print(test_base)
testdata = []
for i in test_base:
    b = bin(i)
    b = b.replace('0b','')
    b = list(map(int,list(b)))
    for j in range(inputsize-len(b)):
        b.insert(0,0)
    for j in range(len(b)):
        if b[j]==0:
            b[j] = 1
        else:
            b[j] = 100
    if i>16:
        testdata.append([[b], [0, 1.0]])
    else:
        testdata.append([[b], [1.0, 0]])
x.stochasticGD(trainingdata,200,0.1,10,testdata = testdata)
x.print_info()
#print(x.feedforward([[1]]))
#print(x.weights)
#x.readweights()
#x.print_info()
#print(x.weights)"""