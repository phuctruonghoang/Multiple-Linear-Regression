import pandas as pd 
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import tkinter as tk
from mpl_toolkits.mplot3d import axes3d
import numpy as np

############ import data ###################
stockData = pd.read_csv(r'C:\Users\HP\Desktop\AI\HistoricalQuotes.csv') #Doc file csv bang padas

df = DataFrame(stockData,columns=['open','high','low','volume','close']) #Tao dataframe chua cac column, row

############ draw diagram ##################

'''
plt.scatter(df['open'], df['close'], color='red')
plt.title('open vs close', fontsize=14)
plt.xlabel('open ($)', fontsize=14)
plt.ylabel('close ($)', fontsize=14)
plt.grid(True)
plt.show()


plt.scatter(df['high'], df['close'], color='blue')
plt.title('high Vs close', fontsize=14)
plt.xlabel('high', fontsize=14)
plt.ylabel('close', fontsize=14)
plt.grid(True)


plt.scatter(df['low'], df['close'], color='green')
plt.title('low Vs close', fontsize=14)
plt.xlabel('low', fontsize=14)
plt.ylabel('close', fontsize=14)
plt.grid(True)


plt.scatter(df['volume'], df['close'], color='yellow')
plt.title('volume Vs close', fontsize=14)
plt.xlabel('volume', fontsize=14)
plt.ylabel('close', fontsize=14)
plt.grid(True)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['open'],df['high'],df['low'],c='blue', marker='o', alpha=0.5)
X = df[['open','close','high']]
Y = df[['open','close','low']]
Z = df[['open','close','volume']]
ax.plot_wireframe(X, Y, Z, rstride = 10, cstride = 10)

plt.show()
'''
	
###### Multiple Linear Regression #### 

def fit(X_matrix,Y_vector): #return values of beta
	X_transpose = np.transpose(X_matrix) #X' (5,130)
	X_mutiply = np.dot(X_transpose,X_matrix) #X'X (5,130)x(130,5) => (5,5)
	X_inverse = np.linalg.inv(X_mutiply) #(X'X)^-1 (5,5)
	XY_multiply = np.dot(X_transpose,Y_vector) #X'Y (5,130) (130,1) => (5,1)
	beta = np.dot(X_inverse,XY_multiply)# (X'X)^-1 X'Y (5,5) (5,1) => (5,1)
	return beta
	
def predict(beta,value): #return predict value
	result = 0
	for i in range(len(beta)):
		result = result + beta[i]*value[i]
	return result

X = df[['open','high','low','volume']] # 4 variable independent
Y = df['close'] # 1 variable dependent
X_new = df[['open','high','low','volume']]

column = pd.Series([])

for i in range(len(X)):
	column[i] = 1

X.insert(0,"one",column) #Them cot dau tien voi gia tri = 1


X_matrix = X[::] #(130,5)
Y_vector =  Y[0:] #(130,1)

beta = fit(X_matrix,Y_vector)

############# sklearn ################
regr = linear_model.LinearRegression()
regr.fit(X_new, Y)

df['predict'] = regr.predict(X_new) #du doan gia tri cua moi bien doc lap

############ statsmodel ###############
X_new = sm.add_constant(X_new)
model = sm.OLS(Y, X_new).fit()
predictions = model.predict(X_new) #tra ve gia tri du doan tuyen tinh tu design matrix

############ tkinter GUI ##############

root = tk.Tk()
root.title('Predict stock price')
canvas1 = tk.Canvas(root, width = 1200, height =480) #tao parent form
canvas1.pack()

interResult = ('Intercept: ',  beta[0]) #lay ket qua intercept
labelInter = tk.Label(root, text = interResult, justify = 'center') #Tao label hien thi ket qua intercept
canvas1.create_window(260, 220, window = labelInter) #Tao window hien thi label

coeffResult = ('Coefficients: ',beta[1:5])
labelCoeff = tk.Label(root, text = coeffResult, justify = 'center')
canvas1.create_window(260, 240, window = labelCoeff)

interceptResult = ('Intercept of sklearn: ', regr.intercept_) #lay ket qua intercept
labelIntercept = tk.Label(root, text = interceptResult, justify = 'center') #Tao label hien thi ket qua intercept
canvas1.create_window(260, 280, window = labelIntercept) #Tao window hien thi label

coefficientsResult = ('Coefficients of sklearn: ', regr.coef_)
labelCoefficientsResult = tk.Label(root, text = coefficientsResult, justify = 'center')
canvas1.create_window(260, 300, window = labelCoefficientsResult)

printModel = model.summary()
labelModel = tk.Label (root, text = printModel, justify = 'center',  relief = 'solid', bg='LightSkyBlue1')
canvas1.create_window(850, 230, window = labelModel)


label1 = tk.Label(root, text = 'open: ')
canvas1.create_window(100, 100, window = label1)

entry1 = tk.Entry(root)
canvas1.create_window(270, 100, window = entry1)

label2 = tk.Label(root, text = 'high: ')
canvas1.create_window(100, 130, window = label2)

entry2 = tk.Entry(root)
canvas1.create_window(270, 130, window = entry2)

label3 = tk.Label(root, text = 'low: ')
canvas1.create_window(98, 160, window = label3)

entry3 = tk.Entry(root)
canvas1.create_window(270, 160, window = entry3)

label4 = tk.Label(root, text = 'volume: ')
canvas1.create_window(109, 190, window = label4)

entry4 = tk.Entry(root)
canvas1.create_window(270, 190, window = entry4)

def getValue(): #lay gia tri dau vao moi de du doan
	global newOpen
	newOpen = float(entry1.get())

	global newHigh
	newHigh = float(entry2.get())

	global newLow
	newLow = float(entry3.get())

	global newVolume
	newVolume = float(entry4.get())

	value = [1,newOpen,newHigh,newLow,newVolume]
	predictResult = ('predict close: ', predict(beta,value))
	lablePredict = tk.Label(root, text = predictResult, bg = 'yellow')
	canvas1.create_window(260, 360, window = lablePredict)

button1 = tk.Button (root, text = 'Predict stock price', command = getValue, bg = 'orange')
canvas1.create_window(270, 400, window = button1)

root.mainloop()



