import torch as T
import torch.nn as nn
import onnx

class Shallow_NET(nn.Module):
    """ A single shallow network for testing ONNX runtime
    """
    def __init__( self, name, n_in, n_out ):
        super(Shallow_NET, self).__init__()
        self.name = name
        self.linear_layer = nn.Linear(n_in, n_out, bias=True)
        self.out_layer = nn.Sigmoid()
        print(self)

    def forward(self, input):
        return self.out_layer(self.linear_layer(input))

## Create a random neural network
n_in  = 77
n_out = 2
net = Shallow_NET( "test_net", n_in, n_out )
dummy_input = T.rand( (1, n_in) )

T.onnx.export(  net,
                dummy_input,
                "dummy_network.onnx",
                verbose=True,
                input_names=["input"],
                output_names=["output"] )

## Load ane print the model
model = onnx.load("dummy_network.onnx")
print( onnx.helper.printable_graph(model.graph) )
