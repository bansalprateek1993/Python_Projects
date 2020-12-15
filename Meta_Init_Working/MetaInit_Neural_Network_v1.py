import torch
from time import time
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms

class MNIST_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784,4)
        self.l2 = nn.Linear(4,10)

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        output = nn.LogSoftmax(dim=1)(self.l2(h1))
        return output

def gradient_quotient(loss, params, eps=1e-5):
    ##Calculating first derivation with respect to all the parametres involved
    grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)

    ## Product Hessian and grad
    prod = torch.autograd.grad(sum([(g ** 2).sum() / 2 for g in grad]),
                               params, retain_graph=True, create_graph=True)

    ## subtracting the gradient with product and dividing it by sum of
    ## gradient and epsilon
    out = sum([((g - p) / (g + eps * (2 * (g >= 0).float() - 1).detach()) \
                - 1).abs().sum() for g, p in zip(grad, prod)])

    ## Dividing it with total number of individual weight/parameters elements to change
    return out / sum([p.data.nelement() for p in params])


def metainit(model, criterion, images, labels, lr=0.1, momentum=0.9, steps=500, eps=1e-5):
    model.eval()
    params = [p for p in model.parameters()
              if p.requires_grad and len(p.size()) >= 2]

    ## Assigning 0 initialized list to memory.
    memory = [0] * len(params)
    for i in range(steps):
        input = images.view(images.shape[0], -1)
        target = labels
        loss = criterion(model(input), target)

        ## Calculating gradient quotient
        gq = gradient_quotient(loss, list(model.parameters()), eps)

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

        for j, p in enumerate(params):
            params[j] = p

    return params

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = datasets.MNIST('Data', download=True, train=True, transform=transform)
    valset = datasets.MNIST('Data', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    model = MNIST_classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    images, labels = next(iter(trainloader))
    ##Using MetaInit##
    images, labels = next(iter(trainloader))
    print("Model Parameters before metaInit = ", list(model.parameters()))

    metainit(model, criterion, images, labels, lr=0.0003, momentum=0.9, steps=500, eps=1e-5)

    print("Model Parameters after metaInit = ", list(model.parameters()))
    ##starting model training After updating model parameters##
    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            # This is where the model learns by backpropagating
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)