import numpy as np

def initialize(layer_dims):
    np.random.seed(1)
    parameters={}
    L=len(layer_dims)

    for l in range(1, L):
        parameters["W"+str(l)]=np.random.randn(layer_dims[l], layer_dims[l-1])/np.sqrt(layer_dims[l-1])
        parameters["B"+str(l)]=np.random.randn(layer_dims[l], 1)
        parameters["G"+str(l)]=np.random.randn(layer_dims[l], 1)
        parameters["BN"+str(l)]=[np.zeros((layer_dims[l], 1)),np.zeros((layer_dims[l], 1)) ]
    return parameters

def sigmoid_forward(Z):
    A= 1/(np.exp(-Z)+1)
    cache=Z
    return A, cache

def relu_forward(Z):
    A= Z*(Z>0)
    cache=Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z= cache
    s=1/(1+np.exp(-Z))
    dZ= dA* s*(1-s)
    return dZ

def relu_backward(dA, cache):
    Z=cache
    dZ=np.array(dA, copy=True)
    dZ[Z<=0]=0
    return dZ
 

def linear_forward(A, W, B, G, BN, mode):
    Z=W@A
    
    if mode=="train":
        mu= np.mean(Z, axis=1, keepdims=True)
        var=np.var(Z, axis=1, keepdims=True)
        BN[0]=0.9*BN[0]+0.1*mu
        BN[1]=0.9*BN[1]+0.1*var
        Z_norm= (Z-mu)/(np.sqrt(var+1e-12))

    elif(mode=="test"):
        Z_norm= (Z-BN[0])/(np.sqrt(BN[1]+1e-12))
    Z_tilda=Z_norm*G+B
    cache=(A, W, B, G, Z, Z_tilda)
    return Z_tilda, cache

def linear_activation_forward(A_prev, W, B, G, activation, BN, mode):
    Z, linear_cache=linear_forward(A_prev, W, B, G, BN, mode)
    if(activation=="sigmoid"):
        A, activation_cache=sigmoid_forward(Z)
    elif(activation=="relu"):
        A, activation_cache=relu_forward(Z)
    cache=(linear_cache, activation_cache)

    return A, cache

def forward(X, parameters, mode):
    L=len(parameters)//4
    A=X
    caches=[]
    for l in range(1, L):
        A_prev=A
        W=parameters["W"+str(l)]
        B=parameters["B"+str(l)]
        G=parameters["G"+str(l)]
        A, cache=linear_activation_forward(A, W, B, G, "relu", parameters["BN"+str(l)], mode)
        caches.append(cache)
    W=parameters["W"+str(L)]
    B=parameters["B"+str(L)]
    G=parameters["G"+str(L)]
    AL, cache=linear_activation_forward(A, W, B, G, "sigmoid", parameters["BN"+str(L)], mode)
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches


def compute_cost(AL, Y, parameters, lambd):
    m = Y.shape[1]
    sum=0
    for l in range(len(parameters)//4):
        sum=sum+np.sum(np.sum(parameters["W"+str(l+1)]**2))
    sum=1./m*sum*lambd
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))+sum
    cost = np.squeeze(cost)  
    assert(cost.shape == ())
    return cost

def linear_backward(dZ_tilda, cache, lambd):
    A_prev, W, B, G, Z, Z_tilda=cache
    m=float(A_prev.shape[1])
    mu=np.mean(Z, axis=1, keepdims=True)
    var=np.var(Z, axis=1, keepdims=True)
    std_inv=1.0/np.sqrt(var+1e-12)
    Z_norm=(Z-mu)*std_inv
    dZ_norm=dZ_tilda*G
    dZ = std_inv * (dZ_norm - np.mean(dZ_norm, axis=1, keepdims=True) - Z_norm * np.mean(dZ_norm * Z_norm, axis=1, keepdims=True))

    dW = 1./m * np.dot(dZ,A_prev.T)+1/m*lambd*W
    
    dG=np.sum(dZ_tilda*Z_norm, axis=1, keepdims=True)

    dB=np.sum(dZ_tilda, axis=1, keepdims=True)

    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, dB, dG

    
def linear_activation_backward(dA, cache,lambd, activation):
    linear_cache, activation_cache=cache
    if(activation=="sigmoid"):
        dZ_tilda=sigmoid_backward(dA, activation_cache)
    if(activation=="relu"):
        dZ_tilda=relu_backward(dA, activation_cache)
    dA_prev, dW, dB, dG=linear_backward(dZ_tilda, linear_cache, lambd)
    return dA_prev, dW, dB, dG

def backward(AL, Y, caches, lambd):
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache=caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["dB" + str(L)], grads["dG" + str(L)] = linear_activation_backward(dAL, current_cache, lambd, activation = "sigmoid")
    for l in reversed(range(L-1)):
        current_cache=caches[l]
        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["dB" + str(l+1)], grads["dG" + str(l+1)] = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,lambd, activation = "relu")
    return grads

def update_parameters(parameters, grads, learning_rate):
    L=len(parameters)//4
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["B"+str(l+1)]=parameters["B"+str(l+1)]-learning_rate*grads["dB"+str(l+1)]
        parameters["G"+str(l+1)]=parameters["G"+str(l+1)]-learning_rate*grads["dG"+str(l+1)]
    return parameters

def predict(X, y, parameters, mode):
    m = X.shape[1]
    n = len(parameters) // 4
    p = np.zeros((1,m))
    probas, caches = forward(X, parameters, mode)
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p
        