import torch
from time import time
from torch import nn, optim
from torch.autograd import Variable


class linear_model(nn.Module):
    def __init__(self,W, b):
        super().__init__()
        self.W = W
        self.b = b
        self.parameters = [self.W,self.b]
        
    def forward(self, x):
        output = W*x + b
        return output


def gradient_quotient(loss, params, eps=1e-5):
    ##Calculating first derivation with respect to all the parametres involved
    grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True, allow_unused=True)
    ## Product Hessian and grad 
    print("grad= ",grad)
    prod = torch.autograd.grad(sum([(g**2).sum() / 2 for g in grad]),
        params, retain_graph=True, create_graph=True)
    
    print("Product of Hessian and grad = ", prod)
    ## subtracting the gradient with product and dividing it by sum of 
    ## gradient and epsilon
    out = sum([((g - p) / (g + eps * (2*(g >= 0).float() - 1).detach()) \
            - 1).abs().sum() for g, p in zip(grad, prod)])
    
    print("Subtracted value =", out)
    ## Dividing it with total number of individual weight/parameters elements to change
    return out / sum([p.data.nelement() for p in params])

def metainit(model, criterion, images, labels, lr=0.1, momentum=0.9, steps=500, eps=1e-5):
#    model.eval()
    params = [p for p in model.parameters
        if p.requires_grad and len(p.size()) >= 2]

    ## Assigning 0 initialized list to memory.
    memory = [0] * len(params)
    input = images
    for i in range(steps):
        target = labels
        loss = criterion(model(input), target)
        
#        print("Loss = %d"% loss)
        ## Calculating gradient quotient
        gq = gradient_quotient(loss, model.parameters, eps)
        
        print("gq", gq)
        ### Optimizing gradient quotient gq functions w.r.t parameter values.
        grad = torch.autograd.grad(gq, params)
        
        ### looping w.r.t to all initial parameter(3 in this case) and grad of GQ
        for j, (p, g_all) in enumerate(zip(params, grad)):
            norm = p.data.norm().item()
            ## Obtaining gradient w.r.t the norm of the parameter w.
            g = torch.sign((p.data * g_all).sum() / norm)
            memory[j] = momentum * memory[j] - lr * g.item()
            new_norm = norm + memory[j]
            p.data.mul_(new_norm / norm)
            p = p.data.mul_(new_norm / norm) 
            
        for j,p in enumerate(params):
            params[j] = p
            
        print("Model Prameters =", model.parameters)
        print("\n")
    return params

#def train_model(optimizer, train_set, train_label):



if __name__ == '__main__':
    W = Variable(torch.randn(1), requires_grad=True)
    b = Variable(torch.randn(1), requires_grad=True)
    x_val = torch.randn(3)
    y_val = torch.randn(3)
    criterion = nn.MSELoss()
    model = linear_model(W, b)
    print(model.parameters)
    updated_param = metainit(model, criterion, x_val, y_val, lr=0.0002, momentum=0.9, steps=500, eps=1e-5)
    print("==========")
    print(updated_param)
    print(model.parameters)
    optimizer = optim.SGD(model.parameters, lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        # Training pass
        optimizer.zero_grad()
        output = model(x_val)
        # print(type(x_val))
        # print(type(y_val))
        # print("output length", len(output))
        # print("y_value length", len(y_val))

        loss = criterion(output, y_val)
        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()
        running_loss += loss.item()
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(x_val)))
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)