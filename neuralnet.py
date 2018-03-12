import numpy as np
import math
import sys
import time

def initial_X_matrix_Y_catogry(filename):
    from numpy import genfromtxt
    read_matrix = genfromtxt(filename, delimiter = ',')
    Y_catogry = []
    Y = []
    for row in read_matrix:
        Y.append(row[0])
        if row[0] not in Y_catogry:
            Y_catogry.append(row[0])
    Y_catogry.sort()

    new_matrix = np.array([[0 for i in range(len(read_matrix[0]))]for j in range(len(read_matrix))],dtype = np.float64)
    for i in range(0,len(new_matrix)):
        for j in range(0,len(read_matrix[0])):
            if j == 0:
                new_matrix[i][j] = 1.0
            else:
                new_matrix[i][j] = read_matrix[i][j]

    return new_matrix,Y_catogry,Y

def initial_alpha_matrix(M,D,init_flag):
    '''
    type should be 1 or 2. 1 stands for RANDOM and 2 stands for ZERO
    '''
    if init_flag == 1:
        alpha = np.array([[0 for i in range(M)] for j in range(D)],dtype = np.float64)
        for i in range(D):
            alpha[i][0:M] = np.random.uniform(-0.1,0.1,M)
    elif init_flag == 2:
        alpha = np.array([[0 for i in range(M)] for j in range(D)],dtype = np.float64)
    return alpha

def compute_z_a(alpha_matrix,x_sample):
    D = len(alpha_matrix)
    #M = len(alpha_matrix[0])
    #a = np.array([0 for i in range(D + 1)],dtype = np.float64)
    z = np.array([0 for i in range(D + 1)],dtype = np.float64)
    for j in range(1,D + 1):
        #for m in range(M):
        z[j] = 1.0/(1.0 +np.exp(-np.dot(alpha_matrix[j-1] , x_sample)))
#    print 'a is',a
   # z = 1.0/(1.0 + np.exp(-a))
    z[0] = 1.0    
    return z

def initialize_beta(z,K):
    beta = np.array([[0 for i in range(len(z))] for j in range (K)], dtype = np.float64)
    return beta

def compute_yhat_b(K,z,beta):
    #beta = np.array([[0 for i in range(len(z))] for j in range (K)], dtype = np.float64)
    #print 'beta is',beta
    b = np.array([0 for i in range(K)], dtype = np.float64)
    for k in range(K):
        #for j in range(len(beta[0])):
        b[k] = np.dot( beta[k] , z)
    #low_item = 0.0
    #for i in b:
    low_item = sum(np.exp(b))
    #y_distribution = np.array([0 for i in range(K)], dtype = np.float64)

    #index = 0
    #for i in b:
    y_distribution = np.exp(b) /low_item
        #index = index + 1
    return y_distribution,b

def SGD(alpha,beta,X,Y,Y_catogry,eita):
    
    #--------------------------------update beta ----------------------------        
    for i_X in range(len(X)):
        #add compute a z here!
        z = compute_z_a(alpha,X[i_X])
        y_distribution, b = compute_yhat_b(len(beta),z,beta)
        
        
        for j in range(len(beta[0])):
            first_item_sum = 0.0
            for k in range(len(beta)):
                first_item = 0.0
                if Y_catogry[k] == Y[i_X]:
                    first_item = y_distribution[k] - 1.0
                else:
                    first_item = y_distribution[k]
                if j != len(beta[0]) - 1:   
                    first_item_sum = first_item_sum + first_item * beta[k][j + 1]   
                #print ('first item is',first_item,'z[j] is',z[j])
                beta[k][j] = beta[k][j] - eita * first_item * z[j]
                #print('new_beta for this round is', new_beta)
    #---------------------------update alpha-------------------------------

            if j != len(beta[0]) - 1:
                alpha[j] = alpha[j] - eita * first_item_sum * z[j + 1] * (1 - z[j + 1]) * X[i_X]
                    #print ('new_alpha for this round is',new_alpha)
                    
                
    return alpha,beta                

def calcu_distribution(alpha,beta,X,Y,Y_catogry):
    y_distribution_matrix = np.array([[0 for i in range(len(Y_catogry))] for j in range(len(X))],dtype = 'float64')
    for i_X in range(len(X)):
        z = compute_z_a(alpha,X[i_X])
        y_distribution, b = compute_yhat_b(len(beta),z,beta)
        y_distribution_matrix[i_X] = y_distribution
    return y_distribution_matrix

def calcu_ACE(X,Y,Y_catogry,y_distribution_matrix):    
    J = 0.0  
    for i in range(len(X)):
        for j in range(len(Y_catogry)):
            if(Y_catogry[j] == Y[i]):
                J = J + math.log(y_distribution_matrix[i][j])
    J_f = -J/len(X) 
    return J_f        

def prediction(y_distribution_matrix,X,Y,Y_catogry,out_filename):
    f = open(out_filename,'w+')
    for i in range(len(y_distribution_matrix)):
        l = list(y_distribution_matrix[i])
        max_value = max(l)
        index = l.index(max_value)
        #print(Y_catogry[index])
        f.write(str(Y_catogry[index])[0]+'\n')
    f.close()    

def compute_errorate(output_file,original_file):
    f_out = open(output_file,'r')
    f_out_read = f_out.readlines()
    f_ori = open(original_file,'r')
    f_ori_read = f_ori.readlines()
    
    count = 0
    for i in range(len(f_out_read)):
        if f_out_read[i][0] != f_ori_read[i][0]:
            count = count + 1
    error_rate = float(count)/float(len(f_out_read))   
    f_out.close()
    f_ori.close()     
    return error_rate    
         
              
if __name__=="__main__":
    time_start=time.time()
    
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    train_out = sys.argv[3]
    validation_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = sys.argv[6]
    hidden_units = sys.argv[7]
    init_flag = sys.argv[8]
    learning_rate = sys.argv[9]
    
    X_tr,Y_catogry_tr,Y_tr = initial_X_matrix_Y_catogry(train_input)
    alpha_tr = initial_alpha_matrix(len(X_tr[0]),int(hidden_units),int(init_flag))
    z_tr = compute_z_a(alpha_tr,X_tr[0])
    beta_tr = initialize_beta(z_tr,len(Y_catogry_tr))
    y_distribution_tr,b_tr = compute_yhat_b(len(Y_catogry_tr),z_tr,beta_tr)
    
    tmp_matrix_tr = []
    tmp_matrix_val = []
    X_val,Y_catogry_val,Y_val = initial_X_matrix_Y_catogry(validation_input)
    alpha_val = initial_alpha_matrix(len(X_val[0]),int(hidden_units),int(init_flag))
    z_val = compute_z_a(alpha_val,X_val[0])
    beta_val = initialize_beta(z_val,len(Y_catogry_val))
    y_distribution_val,b_val = compute_yhat_b(len(Y_catogry_val),z_val,beta_val)
    #print('y_distribution,b is',y_distribution,b )
    f_metrics_out = open(sys.argv[5],'w+')
    f_vali_out = open(sys.argv[4],'w+')
    f_train_out = open(sys.argv[3],'w+')
    
    for i in range(1,int(num_epoch) + 1):
        new_alpha_tr, new_beta_tr = SGD(alpha_tr,beta_tr,X_tr,Y_tr,Y_catogry_tr,float(learning_rate))
        y_distribution_matrix_tr  = calcu_distribution(new_alpha_tr,new_beta_tr,X_tr,Y_tr,Y_catogry_tr)
        alpha_tr = new_alpha_tr
        beta_tr = new_beta_tr
        J_f_tr = calcu_ACE(X_tr,Y_tr,Y_catogry_tr,y_distribution_matrix_tr)
        tmp_matrix_tr = y_distribution_matrix_tr
        f_metrics_out.write('epoch='+str(i) + ' crossentropy(train): '+ str(J_f_tr) + '\n') 
        y_distribution_matrix_val = calcu_distribution(alpha_tr,beta_tr,X_val,Y_val,Y_catogry_val)
        tmp_matrix_val = y_distribution_matrix_val
        J_f_val = calcu_ACE(X_val,Y_val,Y_catogry_val,y_distribution_matrix_val)
        f_metrics_out.write('epoch='+str(i) + ' crossentropy(validation): '+ str(J_f_val) + '\n' )
    prediction(tmp_matrix_tr,X_tr,Y_tr,Y_catogry_tr,train_out)
    prediction(tmp_matrix_val,X_val,Y_val,Y_catogry_tr,validation_out)
    
    error_rate_tr = compute_errorate(train_out,train_input)
    error_rate_val = compute_errorate(validation_out,validation_input)
    f_metrics_out.write('error(train): ' + str(error_rate_tr) + '\n') 
    f_metrics_out.write('error(validation): ' + str(error_rate_val) + '\n') 
    f_metrics_out.close()
    f_vali_out.close()
    f_train_out.close()   
    time_end=time.time() 
    print (time_end-time_start)
