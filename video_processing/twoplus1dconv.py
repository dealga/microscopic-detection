class Conv2Plus1D(keras.layers.Layer):
    def __init_(self, filters, kernel_size, padding):

        ##first apply convolution on the spatial dimensions and then 
        ##apply convolution on the temporal dimension

        ##The output is a feature map that highlights spatial patterns such as edges, textures, and objects.

        ##spatial convolution is like 2d convolution(height and width)

        ##temporal convolution is convolution on 1D(like time) 


        ##The output is a feature map that captures temporal changes, such as motion or sequence dynamics.

        ## using 2+1d convoluton reduces the number of parameters and computation required to 
        ## learn spatiotemporal patterns


        super().__init__()
        self.seq = keras.Sequential([
            layers.Conv3D(filters = filters, kernel_size =(1,kernel_size[1], kernel_size[2]),
                           padding=padding),
            layers.Conv3D(filters = filters, kernel_size =(kernel_size[0],1,1),
                            padding=padding)
        ])

    def call(self, inputs):
        return self.seq(x)
