import matplotlib
matplotlib.use('Agg')
import os

import numpy                        as np
import matplotlib.pyplot            as plt
import tensorflow                   as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras.activations as act
import tensorflow.keras.layers      as layers
import tensorflow.keras.models      as mods

from   tensorflow.keras             import Model
from   model_classes import * 

def templateType(template_type):
	
	templates = {
		0: ("none"       , "none"   ),
		1: ("cnn"        , "ordered"),
		2: ("autoencoder", "ordered")
	}
		
	return templates[template_type]

def returnOrder(layer_genes):
	return layer_genes.ordering_index

def orderByTemplate(order_template, genome):
		
	for layer_index in range(genome.num_layers):
		genome.layer_genes[layer_index].ordering_index = order_template[genome.layer_genes[layer_index].layer_type]
		
	genome.layer_genes.sort(key=returnOrder)
		
	return genome

def calculateNumConvParameters(kernel_sizes, num_input_layers, num_output_layers):
	
	return (np.prod(kernel_sizes)*num_input_layers + 1) * num_output_layers

def calculateEquivilentKernelSize(kernel_size, dilation_size):
	
	return int(kernel_size + (dilation_size - 1)*(kernel_size - 1))

def returnMaxDilationSize(kernel_size, max_size):
	
	return int((max_size - kernel_size) /(kernel_size - 1) + 1)

def calculateConvOuputSize(input_size, kernel_size, stride_size, dilation_size):
	
	kernel_size = calculateEquivilentKernelSize(kernel_size, dilation_size)
	
	return int(((input_size - kernel_size) / stride_size ) + 1)

def calculateConvKernelSize(input_size, output_size, stride_size, dilation_size):
	
	kernel_size = -1*(((output_size - 1) * stride_size) - input_size)
	kernel_size = calculateEquivilentKernelSize(kernel_size, dilation_size)
	
	return int(kernel_size)

def calculateConvTOuputSize(input_size, kernel_size, stride_size, dilation_size):
	
	kernel_size = calculateEquivilentKernelSize(kernel_size, dilation_size)
		
	return int((input_size - 1)*stride_size + kernel_size)

def calculateConvTKernelSize(input_size, output_size, stride_size, dilation_size):
	
	kernel_size = output_size - (input_size - 1)*stride_size
	
	kernel_size = calculateEquivilentKernelSize(kernel_size, dilation_size)
		
	return int(kernel_size)
	
def applyLimits(value, lower_limit, upper_limit, TYPE_MAX):
	
	devisor = (TYPE_MAX * (upper_limit - lower_limit))
	
	new_value = 0
	
	if (devisor > 0):
		new_value = value / devisor
	
	new_value += lower_limit
	
	return int(new_value)			

def addActivation(key):
	#Some of these have options which could be looked into
	activation = {
		 0: act.linear,
		 1: act.elu,
		 2: act.exponential,
		 3: act.hard_sigmoid,
		 4: act.relu,
		 5: act.selu,
		 6: act.sigmoid,
		 7: act.softmax,
		 8: act.softplus,
		 9: act.softsign,
		10: act.swish,
		11: act.tanh,
	}
	return activation[key]
		
def addLayer(genes, input_size):
	
	dimensions = len(input_size)
	
	if ((len(input_size) < 0) or (len(input_size) > 3)):
		print(f"Invalid number of dimensions {len(input_size)}. You probably need to rewrite this code.")
			
	if (genes.activation_present == 0):
		genes.activation_function = 0
	
	#Error checking as both kernel size and dimension cannot be greater than 1:
	if (genes.kernel_stride_x != 1) or (genes.kernel_stride_y != 1):
		genes.kernel_dilation_x = 1;
		genes.kernel_dilation_y = 1;
		
	input_size_x = int(input_size[0])
	input_size_y = 0
	
	if (len(input_size) >= 2):
		input_size_y = int(input_size[1])
		
	#Devide by 0 checks:		
	if (genes.kernel_stride_x <= 0):
		genes.kernel_stride_x = 1;
		
	if (genes.kernel_stride_y <= 0):
		genes.kernel_stride_y = 1;
	
	linear = None
	dense  = None
	conv   = None
	convt  = None
	
	if (genes.layer_type == 0):
		linear = [act.linear, input_size, 0]
	
	elif (genes.layer_type == 1):
		
		total_output_size = genes.dense_output_x * genes.dense_output_y * genes.dense_output_z
		dense =  [
			      layers.Dense(
							   total_output_size,
					           activation = addActivation(genes.activation_function)
		                      ), 
				              (total_output_size,), 
				              total_output_size*sum(input_size) + total_output_size
				 ]
		
	elif (genes.layer_type == 2):
		
		#Error checking to make sure values do not exceed dimensions
		if (genes.kernel_size_x > input_size_x):
			genes.kernel_size_x = input_size_x

		if (calculateEquivilentKernelSize(genes.kernel_size_x, genes.kernel_dilation_x) > input_size_x):
			genes.kernel_dilation_x = int(returnMaxDilationSize(genes.kernel_size_y, input_size_x))	

		if (len(input_size) >= 2):

			if (genes.kernel_size_y > input_size_y):
				genes.kernel_size_y = input_size_y

			if (calculateEquivilentKernelSize(genes.kernel_size_y, genes.kernel_dilation_y) > input_size_y):
				genes.kernel_dilation_y = int(returnMaxDilationSize(genes.kernel_size_y, input_size_y))
		
		conv_dim = {
			0:[act.linear, input_size],
			1:[act.linear, input_size],
			2:[layers.Conv1D(
				genes.num_kernels,
				genes.kernel_size_x,
				strides       = genes.kernel_stride_x,
				dilation_rate = genes.kernel_dilation_x,
				activation    = addActivation(genes.activation_function),
				padding       = "valid"
			 ), 
			  (calculateConvOuputSize(input_size_x, genes.kernel_size_x, genes.kernel_stride_x, genes.kernel_dilation_x), genes.num_kernels),
			   calculateNumConvParameters([genes.kernel_size_x], input_size[-1], genes.num_kernels)
			  ],
			3:[layers.Conv2D(
				genes.num_kernels,
				(genes.kernel_size_x, genes.kernel_size_y),
				strides       = (genes.kernel_stride_x, genes.kernel_stride_y),
				dilation_rate = (genes.kernel_dilation_x, genes.kernel_dilation_y),
				activation    = addActivation(genes.activation_function),
				padding       = "valid"
			 ), 
			   (calculateConvOuputSize(input_size_x, genes.kernel_size_x, genes.kernel_stride_x, genes.kernel_dilation_x), 
				calculateConvOuputSize(input_size_y, genes.kernel_size_y, genes.kernel_stride_y, genes.kernel_dilation_y), 
				genes.num_kernels
			   ),
			   calculateNumConvParameters([genes.kernel_size_x, genes.kernel_size_y], input_size[-1], genes.num_kernels)
			 ]
		}
		
		conv = conv_dim[dimensions]
	elif (genes.layer_type == 3):
		
		convt_dim = {
			0:[act.linear, input_size, 0],
			1:[act.linear, input_size, 0],
			2:[act.linear, input_size, 0],
			3:[layers.Conv2DTranspose(
				genes.num_kernels,
				(genes.kernel_size_x, genes.kernel_size_y),
				strides       = (genes.kernel_stride_x, genes.kernel_stride_y),
				#dilation_rate = (genes.kernel_dilation_x, genes.kernel_dilation_y),
				activation    = addActivation(genes.activation_function),
				padding       = "valid"
			 ), 
			   (calculateConvTOuputSize(input_size_x, genes.kernel_size_x, genes.kernel_stride_x, genes.kernel_dilation_x), 
				calculateConvTOuputSize(input_size_y, genes.kernel_size_y, genes.kernel_stride_y, genes.kernel_dilation_y), 
				genes.num_kernels
			   ),
			   calculateNumConvParameters([genes.kernel_size_x, genes.kernel_size_y], input_size[-1], genes.num_kernels)
			 ]
		}
		
		convt = convt_dim[dimensions]

	types = {
		0: linear,
		1: dense,
        2: conv,
		3: convt
    }
	
	return types[genes.layer_type]

def typeToName(layer_type):
	
	names = {
		0: "Linear",
		2: "Dense",
		3: "Convoloutional",
		4: "Deconvoloutional"
	}
	
	return names[layer_type]

"""
2:layers.Conv1DTranspose(
	genes.num_kernels,
	genes.kernel_size_x,
	strides       = genes.kernel_stride_x,
	dilation_rate = genes.kernel_dilation_x,
	activation    = addActivation(genes.activation_function),
	padding       = "same"
 ),
"""

def addPooling(genes, input_size):
	
	dimensions = len(input_size)
	
	if (genes.pool_size_x > input_size[0]):
		genes.pool_size_x = int(input_size[0])
		
	if (len(input_size) == 3):

		if (genes.pool_size_y > input_size[1]):
			genes.pool_size_y = int(input_size[1])
			
	#Devide by 0 checks:		
	if (genes.pool_stride_x <= 0):
		genes.pool_stride_x = 1;
		
	if (genes.pool_stride_y <= 0):
		genes.pool_stride_y = 1;
		
	if (genes.layer_type != 3):
	
		pool_dim = {
			0: [act.linear, input_size], 
			1: [act.linear, input_size],
			2: [layers.MaxPool1D(
				pool_size = genes.pool_size_x, 
				strides   = genes.pool_stride_y, 
				padding   ='valid' 
			   ),
			   (calculateConvOuputSize(input_size[0], genes.pool_size_x, genes.pool_stride_x, 1), input_size[1]),
			   0
			   ],
			3: [ layers.MaxPool2D(
				pool_size = (genes.pool_size_x  , genes.pool_size_y), 
				strides   = (genes.pool_stride_x, genes.pool_stride_y), 
				padding   ='valid' 
			   ), 
			   (calculateConvOuputSize(input_size[0], genes.pool_size_x, genes.pool_stride_x, 1), 
				calculateConvOuputSize(input_size[1], genes.pool_size_y, genes.pool_stride_y, 1), 
				input_size[2]
			   ),
			   0
			 ]
		}
	
	else:
		
		pool_dim = {
			0: [act.linear, input_size], 
			1: [act.linear, input_size],
			2: [layers.UpSampling1D(
				size = genes.pool_size_x, 
			   ),
			   (calculateConvTOuputSize(input_size[0], genes.pool_size_x, genes.pool_stride_x, 1), input_size[1]),
			   0
			   ],
			3: [ layers.UpSampling2D(
				size = (genes.pool_size_x  , genes.pool_size_y), 
			   ), 
			   (calculateConvTOuputSize(input_size[0], genes.pool_size_x, genes.pool_stride_x, 1), 
				calculateConvTOuputSize(input_size[1], genes.pool_size_y, genes.pool_stride_y, 1), 
				input_size[2]
			   ),
			   0
			 ]
		}
		
	return pool_dim[dimensions]

def addBatchNorm(genes, input_size):

	return [layers.BatchNormalization(), input_size, 0]

def addDropout(genes, input_size):
	
	return [layers.Dropout(genes.dropout_value), input_size, 0]

def addFlatten(genes, input_size):
	
	output_size = 1
		
	for i_size in input_size:
		output_size *= i_size
				
	return [layers.Flatten(), (output_size,), 0]

def popLooseDimensions(old_array):
	
	new_array = []
	
	value_encountered = 0
	for value in np.array(old_array)[::-1]:
		if (value != 1) and (value_encountered == 0):
			new_array.append(value)
			value_encountered = 1
		elif(value_encountered >= 1):
			new_array.append(value)
				
	return list(np.array(new_array)[::-1])


def addReshape(genes, input_size):
	
	return [layers.Reshape(genes.output_shape), genes.output_shape, 0]

def setupCUDA(verbose, device_num):
		
	os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
	
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	if verbose:
		tf.config.list_physical_devices("GPU")