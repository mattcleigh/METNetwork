import torch as T
import torch.nn as nn

class mlp_block(nn.Module):
    def __init__( self, n_in, depth, width, act, nrm, drpt ):
        super(mlp_block, self).__init__()

        block = []
        for d in range(depth):
            inpt = n_in if d==0 else width
            block.append( nn.Linear( inpt, width ) )
            if act is not None: block.append( act )
            if drpt>0:          block.append( nn.Dropout( p=drpt ) )
            if nrm:             block.append( nn.LayerNorm( width ) )
        self._ = nn.Sequential(*block)

    def forward(self, input):
        return self._(input)

class res_mlp_block(mlp_block):
    def __init__( self, *args ):
        super(res_mlp_block, self).__init__( *args )

    def forward(self, input):
        return self._(input) + input

def res_mlp_creator( n_in=1, n_out=None, depth=1, width=64, skips=1,
                     act_h=nn.ReLU(), act_o=None, nrm=False, drpt=0 ):

    layers = []

    ## Create the input layer block
    layers.append( mlp_block( n_in, 1, width, act_h, nrm, drpt ) )

    ## Create each hidden layer block
    l=1
    while l < depth:
        if skips>0 and l+skips <= depth:
            layers.append( res_mlp_block( width, skips, width, act_h, nrm, drpt ) )
            l += skips
        else:
            layers.append( mlp_block( width, 1, width, act_h, nrm, drpt ) )
            l += 1


    ## Create the output block
    layers.append( mlp_block( width, 1, n_out, act_o, False, 0 ) )

    return nn.Sequential( *layers )
