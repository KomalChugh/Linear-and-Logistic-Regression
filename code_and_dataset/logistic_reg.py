import random
import math
import numpy as np

# standardises the training data and stores the mean and standard deviation of each attribute and returns the training matrix X and true output Y
def standardise_train_data():
	attr_list = []
	train_array = np.zeros([len(train_data),3])
	train_output_list = []
	
	for i in train_data:
		data_list = []
		att = i.split(",")
		for j in range(len(att)-1):
			data_list.append(float(att[j]))
			mean_list[j] = mean_list[j] + float(att[j])
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
	
	
	for i in range(len(train_data)):
		train_array[i,0] = 1.0
		for j in range(1,3):
			train_array[i,j] = float((attr_list[i][j-1]-mean_list[j-1])/std_dev_list[j-1])
			
	train_output = np.array(train_output_list)
	train_output = train_output.reshape(-1, 1)
	return train_array, train_output

# standardises the test data based on mean and standard deviation according to training set and returns the input matrix X and true output Y
def standardise_test_data():
	attr_list = []
	test_array = np.zeros([len(test_data),3])
	test_output_list = []
	
	for i in test_data:
		data_list = []
		att = i.split(",")
		for j in range(len(att)-1):
			data_list.append(float(att[j]))
		attr_list.append(data_list)
		test_output_list.append(int(att[len(att)-1]))
				
	
	for i in range(len(test_data)):
		test_array[i,0] = 1.0
		for j in range(1,3):
			test_array[i,j] = float((attr_list[i][j-1]-mean_list[j-1])/std_dev_list[j-1])


	test_output = np.array(test_output_list)
	test_output = test_output.reshape(-1, 1)
	return test_array, test_output

# computes error	
def compute_err(Y, Ydash):
	err = 0.0
	for i in range(len(Y)):
		if(Ydash[i,0]<=0.00000001):
			a = 0
		else:
			a = math.log(Ydash[i,0])
		if(1-Ydash[i,0]<=0.000000001):
			b = 0
		else:
			b = math.log(1.0-Ydash[i,0])
		err = err - float((Y[i,0]*a)+((1-Y[i,0])*b))
	return err


# logistic regression using gradient descent
def mylogisridgereg(X, Y, lambda_parameter,feature_transform):
	W = np.random.uniform(low=-0.1,high=0.1,size=(len(X[0]),1))
	FX = mylogisridgeregeval(X, W)
	alpha = 0.001
	W_new = W
	W_old = -1 * np.ones([len(X[0]),1])
	err_new = compute_err(Y, mylogisridgeregeval(X, W_new))
	err_old = compute_err(Y, mylogisridgeregeval(X, W_old))
	i = 0
	while(err_new!=err_old):
		if(feature_transform=="true" and i==1000):
			break
		if(err_new==0):
			break
		W_old = W_new
		FX = mylogisridgeregeval(X, W_old)
		err_old = compute_err(Y, mylogisridgeregeval(X, W_old))
		W_new = W_old - alpha * np.add((2*lambda_parameter*W),np.dot(X.transpose(),np.subtract(FX,Y)))
		err_new = compute_err(Y, mylogisridgeregeval(X, W_new))
		i = i+1
		

	return W_new

# logistic regression using newton raphson	
def newton_raphson(X, Y, lambda_parameter,feature_transform):
	W = np.random.uniform(low=-0.1,high=0.1,size=(len(X[0]),1))
	W_new = W
	W_old = -1 * np.ones([len(X[0]),1])
	err_new = compute_err(Y, mylogisridgeregeval(X, W_new))
	err_old = compute_err(Y, mylogisridgeregeval(X, W_old))
	I = np.identity(len(X[0]))
	
	k = 0
	while(err_new!=err_old):
		if(err_new==0):
			break
		if(feature_transform=="true" and k==1000):
			break
		W_old = W_new
		FX = mylogisridgeregeval(X, W_old)
		err_old = compute_err(Y, mylogisridgeregeval(X, W_old))
		R = np.zeros([len(X),len(X)])
		for i in range(len(R)):
			R[i,i] = FX[i,0]*(1.0-FX[i,0])
		A = np.linalg.inv(np.add(np.dot(np.dot(X.transpose(),R),X),(2*lambda_parameter*I)))
		B = np.add((2*lambda_parameter*W),np.dot(X.transpose(),np.subtract(FX,Y)))
		W_new = W_old - np.dot(A,B)
		err_new = compute_err(Y, mylogisridgeregeval(X, W_new))
		k = k+1
		
		
	return W_new
		

# computes FX for given set of X and W	
def mylogisridgeregeval(X, W):
	FX = X.dot(W)
	FX = FX.astype(np.float128)
	FX = (np.exp(-1*FX))+1
	FX = 1.0 / FX
	return FX

# performs thresholding on the output and computes accuracy	
def predict_output(FX,Y):
	wrong_count = 0
	output = np.zeros([len(FX),len(FX[0])])
	for i in range(len(FX)):
		if(FX[i,0]>=0.5):
			output[i,0]=1
		else:
			output[i,0]=0
		if(output[i,0]!=Y[i,0]):
			wrong_count = wrong_count + 1
			
	acc = ((len(Y)-wrong_count)*100)/len(Y)
	return output,acc
	

# transforms the given matrix X into high dimensional attributes governed by given degree and then standardises the high dimensional matrix	
def featuretransform(X, degree):
	exponent_matrix = np.zeros([degree+1,degree+1])
	total_att = 0
	for i in range(degree+1):
		for j in range(degree+1):
			if(j<=degree-i):
				exponent_matrix[i,j]=1
				total_att = total_att + 1
				
	X_highdegree = np.ones([len(X),total_att])
	att = 0
	mean = [0.0] * total_att
	std_dev = [0.0] * total_att
	for i in range(degree+1):
		for j in range(degree+1):
			if(exponent_matrix[i,j]==1):
				for k in range(len(X)):
					X_highdegree[k,att] = float(pow(X[k,0],i) * pow(X[k,1],j))
					mean[att] = mean[att] + X_highdegree[k,att]					
				att = att + 1
	
				
	for i in range(len(mean)):
		mean[i] = float(mean[i] / len(X))
		
	
	for j in range(len(std_dev)):
		for k in range(len(X)):
			std_dev[j] = std_dev[j] + ((X_highdegree[k,j]-mean[j]) * (X_highdegree[k,j]-mean[j]))		
		
	for i in range(len(std_dev)):
		std_dev[i] = float(std_dev[i] / len(X))
		std_dev[i] = math.sqrt(std_dev[i])
		
	for i in range(len(X_highdegree)):
		for j in range(len(X_highdegree[0])):
			if(j==0):
				continue
			X_highdegree[i,j] = float((X_highdegree[i,j]-mean[j])/std_dev[j])
		
	return X_highdegree
			
# test set is 20% and train set is 80% of the total dataset
dataset_file = list(open("l2/credit.txt"))
test_data_size = int(0.2 * len(dataset_file))
selected_data = [0] * len(dataset_file)
test_data = []
train_data = []
i = 0
while(1):
	if(i==test_data_size):
		break
	j = random.randint(0,len(dataset_file)-1)
	if(selected_data[j]==0):
		i = i+1
		selected_data[j]=1
		test_data.append(dataset_file[j])
		

for j in range(len(selected_data)):
	if(selected_data[j]==0):
		selected_data[j]=1
		train_data.append(dataset_file[j])
	
mean_list = [0.0] * 2
std_dev_list = [0.0] * 2

# standardise both training and test datasets
train_X,train_Y = standardise_train_data()
test_X,test_Y = standardise_test_data()
lambda_parameter = [0.01,0.1,1,2,5]
degree = 6  # optimal degree 


for m in range(len(lambda_parameter)):


	print '%-50s%-5s' %("lambda",lambda_parameter[m])
	print("\nWithout feature transformation\n")
	# Train dataset
	trainWGrad = mylogisridgereg(train_X,train_Y,lambda_parameter[m],"false")
	trainFXGrad = mylogisridgeregeval(train_X, trainWGrad)
	trainFXGrad,trainAccGrad = predict_output(trainFXGrad,train_Y)
	print '%-50s%-5s' %("Gradient descent train set accuracy",trainAccGrad)
	

	trainWNewton = newton_raphson(train_X,train_Y,lambda_parameter[m],"false")
	trainFXNewton = mylogisridgeregeval(train_X, trainWNewton)
	trainFXNewton,trainAccNewton = predict_output(trainFXNewton,train_Y)
	print '%-50s%-5s' %("Newton Raphson train set accuracy",trainAccNewton)
	

	# Test dataset
	testWGrad = mylogisridgereg(test_X,test_Y,lambda_parameter[m],"false")
	testFXGrad = mylogisridgeregeval(test_X, testWGrad)
	testFXGrad,testAccGrad = predict_output(testFXGrad,test_Y)
	print '%-50s%-5s' %("Gradient descent test set accuracy",testAccGrad)

	testWNewton = newton_raphson(test_X,test_Y,lambda_parameter[m],"false")
	testFXNewton = mylogisridgeregeval(test_X, testWNewton)
	testFXNewton,testAccNewton = predict_output(testFXNewton,test_Y)
	print '%-50s%-5s' %("Newton Raphson test set accuracy",testAccNewton)
	
	
	# Feature transformation   ---->  Train dataset
	print("\nWith feature transformation\n")
	train_low_degree_X = np.zeros([len(train_data),2])
	for i in range(len(train_data)):
		att = train_data[i].split(",")
		for j in range(len(att)-1):
			train_low_degree_X[i,j] = float(att[j])
	
	train_high_degree_X = featuretransform(train_low_degree_X,degree)

	# Gradient Descend
	trainW_grad = mylogisridgereg(train_high_degree_X,train_Y,lambda_parameter[m],"true")
	trainFX_grad = mylogisridgeregeval(train_high_degree_X, trainW_grad)
	trainFX_grad,trainAccGrad = predict_output(trainFX_grad,train_Y)

	# Newton Raphson
	trainW_newton = newton_raphson(train_high_degree_X,train_Y,lambda_parameter[m],"true")
	trainFX_newton = mylogisridgeregeval(train_high_degree_X, trainW_newton)
	trainFX_newton,trainAccNewton = predict_output(trainFX_newton,train_Y)
	
	print '%-50s%-5s' %("Gradient descent train set accuracy",trainAccGrad)
	print '%-50s%-5s' %("Newton Raphson train set accuracy",trainAccNewton)
	
	# Feature transformation   ---->  Test dataset
	test_low_degree_X = np.zeros([len(test_data),2])
	for i in range(len(test_data)):
		att = test_data[i].split(",")
		for j in range(len(att)-1):
			test_low_degree_X[i,j] = float(att[j])
	
	test_high_degree_X = featuretransform(test_low_degree_X,degree)

	# Gradient Descend
	testW_grad = mylogisridgereg(test_high_degree_X,test_Y,lambda_parameter[m],"true")
	testFX_grad = mylogisridgeregeval(test_high_degree_X, testW_grad)
	testFX_grad,testAccGrad = predict_output(testFX_grad,test_Y)

	# Newton Raphson
	testW_newton = newton_raphson(test_high_degree_X,test_Y,lambda_parameter[m],"true")
	testFX_newton = mylogisridgeregeval(test_high_degree_X, testW_newton)
	testFX_newton,testAccNewton = predict_output(testFX_newton,test_Y)

	print '%-50s%-5s' %("Gradient descent test set accuracy",testAccGrad)
	print '%-50s%-5s' %("Newton Raphson test set accuracy",testAccNewton)
	print("\n\n\n\n\n")
	



