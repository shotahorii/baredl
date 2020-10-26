import baredl.functions as F
import baredl.layers as L


class Model(L.Layer):
    pass


class Sequential(Model):
    def __init__(self, *layers):
        if not layers:
            raise ValueError('At least one layer needed.')
        elif not all([isinstance(l, Layer) for l in layers]):
            raise ValueError('Every input needs to be a Layer instance.')

        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)
            
    def forward(self, x):  
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)