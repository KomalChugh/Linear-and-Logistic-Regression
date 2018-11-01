import random
import math
import numpy as np

# sample the data other than test data into training and validation set based on the value of fraction
def sample_data(frac,size):
	train_data_size = int(frac * size)
	i = 0
	while(1):
		if(i==train_data_size):
			break
		j = random.randint(0,len(dataset_file)-1)
		if(new_selected_data[j]==0):
			i = i+1
			new_selected_data[j]=1
			train_data.append(dataset_file[j])
			
	for i in range(len(dataset_file)):
		if(new_selected_data[i]==0):
			new_selected_data[i]=1
			validation_data.append(dataset_file[i])
		

# standardises the training data and stores the mean and standard deviation of each attribute and returns the training matrix X and true output Y
def standardise_train_data():
	attr_list = []
	gender_list = []
	train_array = np.zeros([len(train_data),11])
	train_output_list = []
	
	for i in train_data:
		data_list = []
		att = i.split(",")
		gender_list.append(att[0])
		for j in range(1,len(att)-1):
			data_list.append(float(att[j]))
			mean_list[j-1] = mean_list[j-1] + float(att[j])
		attr_list.append(data_list)
		train_output_list.append(int(att[len(att)-1]))
				
	for i in range(len(mean_list)):
		mean_list[i] = float(mean_list[i] / len(train_data))
		
	
	for list1 in attr_list:
		for j in range(len(list1)):
			std_dev_list[j] = std_dev_list[j] + ((list1[j]-mean_list[j]) * (list1[j]-mean_list[j]))		
		
	for i in range(len(std_dev_list)):
		std_dev_list[i] = float(std_dev_list[i] / len(train_data))
		std_dev_list[i] = math.sqrt(std_dev_list[i])
	
	# append 1 at the beginning and standardise input attributes leaving gender
	for i in range(len(train_data)):
		train_array[i,0] = 1.0
		if(gender_list[i]=="M"):
			train_array[i,3] = 1.0
		elif(gender_list[i]=="I"):
			train_array[i,2] = 1.0	
		elif(gender_list[i]=="F"):
			train_array[i,1] = 1.0
		for j in range(4,11):
			train_array[i,j] = float((attr_list[i][j-4]-mean_list[j-4])/std_dev_list[j-4])
			
	train_output = np.array(train_output_list)
	train_output = train_output.reshape(-1, 1)
	return train_array, train_output

# standardises the test data based on mean and standard deviation according to training set and returns the input matrix X and true output Y
def standardise_test_data():
	attr_list = []
	gender_list = []
	test_array = np.zeros([len(test_data),11])
	test_output_list = []
	
	for i in test_data:
		data_list = []
		att = i.split(",")
		gender_list.append(att[0])
		for j in range(1,len(att)-1):
			data_list.append(float(att[j]))
		attr_list.append(data_list)
		test_output_list.append(int(att[len(att)-1]))
				
	
	for i in range(len(test_data)):
		test_array[i,0] = 1.0
		if(gender_list[i]=="M"):
			test_array[i,3] = 1.0
		elif(gender_list[i]=="I"):
			test_array[i,2] = 1.0	
		elif(gender_list[i]=="F"):
			test_array[i,1] = 1.0
		for j in range(4,11):
			test_array[i,j] = float((attr_list[i][j-4]-mean_list[j-4])/std_dev_list[j-4])


	test_output = np.array(test_output_list)
	test_output = test_output.reshape(-1, 1)
	return test_array, test_output

# computes mean squared error between true and predicted values
def meansquarederr(Y, Ydash):
	D = np.subtract(Ydash,Y)
	err =  ((np.dot(D.transpose(),D))/len(Y))[0,0]
	return err


# performs linear regression using gradient descent and returns weight matrix W and the stopping criteria is the number of iterations
def mylinridgereg(X, Y, lambda_parameter):
	W = np.random.rand(len(X[0]),1)
	FX = X.dot(W)
	alpha = 0.0001
	W_new = W
	W_old = -1 * np.ones([len(X[0]),1])
	err_new = meansquarederr(Y, X.dot(W_new))
	err_old = meansquarederr(Y, X.dot(W_old))
	m = 0
	
	while(1):
		
		if(m==1000):
			break
		W_old = W_new
		FX = X.dot(W_old)
		err_old = meansquarederr(Y, X.dot(W_old))
		W_new = W_old - alpha * np.add((2*lambda_parameter*W),np.dot(X.transpose(),np.subtract(FX,Y)))
		err_new = meansquarederr(Y, X.dot(W_new))
		m = m+1

	
	return W_new

# computes FX = X.W	
def mylinridgeregeval(X, weights):
	return np.round(X.dot(weights))

# test set is fixed to 20% of the total dataset and is choosen randomly	
dataset_file = list(open("l2/linregdata"))
test_data_size = int(0.2 * len(dataset_file))
selected_data = [0] * len(dataset_file)
test_data = []
i = 0
while(1):
	if(i==test_data_size):
		break
	j = random.randint(0,len(dataset_file)-1)
	if(selected_data[j]==0):
		i = i+1
		selected_data[j]=1
		test_data.append(dataset_file[j])
		

# frac denotes the fraction of training set from the remaining examples
frac = [0.03,0.1,0.15,0.25,0.38,0.55,0.7,0.8]
lambdas = [1, 2, 5, 8, 10, 14, 17, 21, 25, 30, 36, 42, 54, 67, 79, 83, 96]


for a in range(len(frac)):
	print '%-12s%-10s%-15s%-15s' %("fraction","lambda","avg train err","avg test err")
	for j in range(len(lambdas)):
		avg_train_err = 0.0
		avg_test_err = 0.0
		lambda_parameter = lambdas[j]
		
		# for each combination of training set fraction and lambda, average of error is taken over 100 iterations
		for k in range(100):
			train_data = []
			validation_data = []
	
			new_selected_data = [0] * len(dataset_file)
			for i in range(len(new_selected_data)):
				if(selected_data[i]==1):
					new_selected_data[i]=1
		
			sample_data(frac[a],int(0.8 * len(dataset_file)))
	
			mean_list = [0.0] * 7
			std_dev_list = [0.0] * 7

			# standardise both training and test datasets
			train_X,train_Y = standardise_train_data()
			test_X,test_Y = standardise_test_data()
	
		
			# compute W and FX for training set
			W = mylinridgereg(train_X, train_Y, lambda_parameter)
			train_FX = mylinridgeregeval(train_X, W)

			# compute error
			train_err = meansquarederr(train_Y, train_FX)
			train_err = round(train_err,2)
			avg_train_err = avg_train_err + train_err
		
			# compute FX for test set
			test_FX = mylinridgeregeval(test_X, W)

			# compute error
			test_err = meansquarederr(test_Y, test_FX)
			test_err = round(test_err,2)
			avg_test_err = avg_test_err + test_err
			
		avg_test_err = avg_test_err/100.0
		avg_train_err = avg_train_err/100.0
		avg_test_err = round(avg_test_err,4)
		avg_train_err = round(avg_train_err,4)
		
		print '%-12s%-10s%-15s%-15s' %(frac[a],lambda_parameter,avg_train_err,avg_test_err)
	
	
	print("\n\n\n")

