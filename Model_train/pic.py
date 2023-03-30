import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def loadDatadet(infile):
    f = open(infile,'r')
    sourceInLine = f.readlines()
    dataset=[]
    for line in sourceInLine:
        temp1 = line.strip()
        temp1 = temp1.replace('[', '')
        temp1 = temp1.replace(']', '')
        temp2 = temp1.split(', ')
        for i in range(1,len(temp2)):
           temp2[i] = float(temp2[i])
           dataset.append(temp2[i])
    return dataset


y1 = loadDatadet("ou1.txt")
y2 = loadDatadet("ou2.txt")
y3 = loadDatadet("ou3.txt")
print(y1)
print(y2)


y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
x = range(0, len(y1), 1)
xmax = np.argmax(x)
y1max = np.max(y1)
y1min = np.min(y1)
y2max = np.max(y2)
y2min = np.min(y2)

x_major_locator = MultipleLocator(2)
y_major_locator = MultipleLocator(0.05)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize = (10,5))
plt.title('Test_result')
plt.xlabel("Number", fontsize = 20)
plt.ylabel('Reward', fontsize = 25)
plt.xlim(-0.5, xmax + 0.5)
plt.ylim(-0.1,0.2)
plt.plot(x, y1, color = 'blue', linewidth = 1, linestyle = '-', label = 'alr=0.01 clr=0.001 batchsize=32', marker = 'o')
#plt.plot(x, y2, color = 'red', linewidth = 1, linestyle = '-', label = 'alr=0.01 clr=0.001 batchsize=64', marker = '*')
#plt.plot(x, y3, color = 'orange', linewidth = 1, linestyle = '-', label = 'alr=0.001 clr=0.0001 batchsize=32', marker = 'x')
plt.legend(loc='lower right', bbox_to_anchor=(0.9, 0.2))

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.savefig("./picture/test3.png")
plt.show()