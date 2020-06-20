
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
'''
Possible adversarial approach with network (2)
    Creates representative for class that performs worst on validation
Batch normalize weights in fc_network

Ensemble method with model trained via Low LR and High LR
    Advantage with low lr is greater generalization
    Advantage with high lr is greater accuracy
    (possibly compute similarity of input to training data
        for weighting between the two models)
'''
class NN(nn.Module):
    
    def __init__(self, fc_depth=1, hidden_units = 128, input_dim=5, out_dim=1):
        super().__init__()

        #self.conv_param = conv_param
        self.repr_dim = hidden_units
        self.fc_depth = fc_depth
        
        #_, H, W = self.add_convolutions()
        #self.conv_res_flat = H*W
        output_dim = out_dim
        
        if fc_depth <= 1:
            self.fc1 = nn.Linear(input_dim, output_dim)
        
        elif fc_depth == 2:
            self.fc1 = nn.Linear(input_dim, self.repr_dim)# best at 512
            self.fc2 = nn.Linear(self.repr_dim, output_dim)
            

        
        else: # fc_depth > 2:
            self.fc1 = nn.Linear(input_dim, self.repr_dim)
            
            for n in range(2, fc_depth):
                exec(f'self.fc{n} = nn.Linear(self.repr_dim, self.repr_dim)')
            exec(f'self.fc{self.fc_depth} = nn.Linear(self.repr_dim, {output_dim})')
            #self.fc_last = nn.Linear(self.repr_dim, 5)
        #

        self.init_weights()


    def init_weights(self):
        '''
        conv = self.conv1
        C_in = conv.weight.size(0)
        nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5**5*C_in))
        nn.init.constant_(conv.bias, 0.0)
        '''
        if self.fc_depth == 1:
            fc = self.fc1
            C_in = fc.weight.size(1) # size = (out_dim, in_dim) 
            nn.init.normal_(fc.weight, 0.0, 2 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)
        else:
            for n in range(1, self.fc_depth+1):
                fc = eval(f'self.fc{n}')
                C_in = fc.weight.size(1)
                nn.init.normal_(fc.weight, 0.0, 1 / sqrt(C_in))
                nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        #print(x.shape)
        #N, W = x.shape
        #z = x.view(-1, W)
        # TODO: Batching
        #print(x)
        z = x
        #z = self.apply_convs(x)
        #z = z.view(-1, self.conv_res_flat)
        
        for n in range(1, self.fc_depth+1):
            z = eval(f'self.fc{n}')(z)
            z = F.relu(z)
        

        return z
    
if __name__ == "__main__":
    in_dim = 5
    m = NN(input_dim= in_dim)
    x =  torch.randn(3, in_dim)
    
    z = m.forward(x)
    print(z.shape)
    
