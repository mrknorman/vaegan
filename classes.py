import matplotlib
matplotlib.use('Agg')

import tensorflow                   as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy                        as np

from   tensorflow.keras             import Model
from   model_functions              import *

class layer_genes_s():
	
	ordering_index = None
	output_shape   = None

	def __init__(self, TYPE_MAX, apply_limits = True, num_layer_genes = None, layer_genes = None, codec = None):

		self.layer_genes         = layer_genes
		
		num_layer_genes = codec.num_layer_genes;

		layer_gene_index = 0

		if (not self.testVariableCount(num_layer_genes)):
			print(f"Error! Class model layer: {num_layer_genes} does not have the correct number of variables: {len(self.layer_genes)}, exiting!")
			quit(1)
		else:

			self.layer_type          = self.layer_genes[layer_gene_index]; layer_gene_index += 1;

			self.activation_present  = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.activation_function = self.layer_genes[layer_gene_index]; layer_gene_index += 1;

			self.dense_output_x      = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.dense_output_y      = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.dense_output_z      = self.layer_genes[layer_gene_index]; layer_gene_index += 1;

			self.num_kernels         = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.kernel_size_x       = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.kernel_size_y       = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.kernel_stride_x     = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.kernel_stride_y     = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.kernel_dilation_x   = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.kernel_dilation_y   = self.layer_genes[layer_gene_index]; layer_gene_index += 1;

			self.pool_present        = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.pool_type           = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.pool_size_x         = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.pool_size_y         = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.pool_stride_x       = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.pool_stride_y       = self.layer_genes[layer_gene_index]; layer_gene_index += 1;

			self.batch_norm_present  = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.dropout_present     = self.layer_genes[layer_gene_index]; layer_gene_index += 1;
			self.dropout_value       = self.layer_genes[layer_gene_index] / TYPE_MAX ; layer_gene_index += 1;
						
			if (not self.testVariableCountEnd(layer_gene_index, num_layer_genes)):
				print(f"Error! Class model layer: {layer_gene_index}, does not have the correct number of variables: {num_layer_genes}, exiting!")
				quit(1)
			
			if ((apply_limits == True) and (codec != None)):
			
				self.layer_type          = applyLimits(self.layer_type         , codec.layer_type_limits[0]         , codec.layer_type_limits[1]         , TYPE_MAX)
				
				self.activation_present  = applyLimits(self.activation_present , codec.activation_present_limits[0] , codec.activation_present_limits[1] , TYPE_MAX)
				self.activation_function = applyLimits(self.activation_function, codec.activation_function_limits[0], codec.activation_function_limits[1], TYPE_MAX)		
				
				self.dense_output_x      = applyLimits(self.dense_output_x     , codec.dense_output_x_limits[0]     , codec.dense_output_x_limits[1]     , TYPE_MAX)
				self.dense_output_y      = applyLimits(self.dense_output_y     , codec.dense_output_y_limits[0]     , codec.dense_output_y_limits[1]     , TYPE_MAX)
				self.dense_output_z      = applyLimits(self.dense_output_z     , codec.dense_output_z_limits[0]     , codec.dense_output_z_limits[1]     , TYPE_MAX)
				
				self.num_kernels         = applyLimits(self.num_kernels        , codec.num_kernels_limits[0]        , codec.num_kernels_limits[1]        , TYPE_MAX)
				self.kernel_size_x       = applyLimits(self.kernel_size_x      , codec.kernel_size_x_limits[0]      , codec.kernel_size_x_limits[1]      , TYPE_MAX)
				self.kernel_size_y       = applyLimits(self.kernel_size_y      , codec.kernel_size_y_limits[0]      , codec.kernel_size_y_limits[1]      , TYPE_MAX)
				self.kernel_stride_x     = applyLimits(self.kernel_stride_x    , codec.kernel_stride_x_limits[0]    , codec.kernel_stride_x_limits[1]    , TYPE_MAX)
				self.kernel_stride_y     = applyLimits(self.kernel_stride_y    , codec.kernel_stride_y_limits[0]    , codec.kernel_stride_y_limits[1]    , TYPE_MAX)
				self.kernel_dilation_x   = applyLimits(self.kernel_dilation_x  , codec.kernel_dilation_x_limits[0]  , codec.kernel_dilation_x_limits[1]  , TYPE_MAX)
				self.kernel_dilation_y   = applyLimits(self.kernel_dilation_y  , codec.kernel_dilation_y_limits[0]  , codec.kernel_dilation_y_limits[1]  , TYPE_MAX)
				
				self.pool_present        = applyLimits(self.pool_present       , codec.pool_present_limits[0]       , codec.pool_present_limits[1]       , TYPE_MAX)
				self.pool_type           = applyLimits(self.pool_type          , codec.pool_type_limits[0]          , codec.pool_type_limits[1]          , TYPE_MAX)
				self.pool_size_x         = applyLimits(self.pool_size_x        , codec.pool_size_x_limits[0]        , codec.pool_size_x_limits[1]        , TYPE_MAX)
				self.pool_size_y         = applyLimits(self.pool_size_y        , codec.pool_size_y_limits[0]        , codec.pool_size_y_limits[1]        , TYPE_MAX)
				self.pool_stride_x       = applyLimits(self.pool_stride_x      , codec.pool_stride_x_limits[0]      , codec.pool_stride_x_limits[1]      , TYPE_MAX)
				self.pool_stride_y       = applyLimits(self.pool_stride_y      , codec.pool_stride_y_limits[0]      , codec.pool_stride_y_limits[1]      , TYPE_MAX)
				
				self.batch_norm_present  = applyLimits(self.batch_norm_present , codec.batch_norm_present_limits[0] , codec.batch_norm_present_limits[1] , TYPE_MAX)
				self.dropout_present     = applyLimits(self.dropout_present    , codec.dropout_present_limits[0]    , codec.dropout_present_limits[1]    , TYPE_MAX)

	def testVariableCount(self, num_layer_genes):
		return_value = 0

		if (num_layer_genes == len(self.layer_genes)):
			return_value = 1
	
		return return_value
	
	def testVariableCountEnd(self, layer_gene_index, num_layer_genes):
		return_value = 0
		
		if (layer_gene_index == layer_gene_index):
			return_value = 1
			
		return return_value

class genome_s():

	def __init__(self, TYPE_MAX, genome = None, codec = None, apply_limits = True):

		if codec != None:

			self.genome = genome
			
			genome_index = 0

			self.num_layers                  = self.genome[genome_index]; genome_index += 1
			self.input_alignment             = self.genome[genome_index]; genome_index += 1
			self.output_alignement           = self.genome[genome_index]; genome_index += 1

			self.optimizer_type              = self.genome[genome_index]; genome_index += 1
			self.loss_type                   = self.genome[genome_index]; genome_index += 1
			
			self.learning_rate               = self.genome[genome_index]/TYPE_MAX; genome_index += 1
			self.batch_size                  = self.genome[genome_index]; genome_index += 1

			self.num_epochs                   = self.genome[genome_index]; genome_index += 1
			self.num_semesters               = self.genome[genome_index]; genome_index += 1

			self.layer_genes                 = [[]]*self.num_layers  

			if (apply_limits == True):
										
				self.num_layers              = applyLimits(self.num_layers, codec.num_layers_limits[0], codec.num_layers_limits[1], TYPE_MAX)
							
				self.optimizer_type          = applyLimits(self.optimizer_type, codec.optimizer_type_limits[0], codec.optimizer_type_limits[1], TYPE_MAX)
				self.loss_type               = applyLimits(self.loss_type     , codec.loss_type_limits[0]     , codec.loss_type_limits[1]     , TYPE_MAX)

				self.learning_rate           = applyLimits(self.learning_rate , codec.learning_rate_limits[0] , codec.learning_rate_limits[1] , TYPE_MAX)
				self.batch_size              = applyLimits(self.batch_size    , codec.batch_size_limits[0]    , codec.batch_size_limits[1]    , TYPE_MAX)
				
				self.num_epochs               = applyLimits(self.num_epochs     , codec.num_epochs_limits[0]     , codec.num_epochs_limits[1]     , TYPE_MAX)
				self.num_semesters           = applyLimits(self.num_semesters , codec.num_semesters_limits[0] , codec.num_semesters_limits[1] , TYPE_MAX)
				
			for layer_idx in range(self.num_layers):

				layer_index_end   = genome_index + codec.num_layer_genes
				
				layer_genes = self.genome[genome_index:layer_index_end]

				genome_index += codec.num_layer_genes 

				self.layer_genes[layer_idx] = layer_genes_s(TYPE_MAX, apply_limits = apply_limits, num_layer_genes = codec.num_layer_genes, layer_genes = layer_genes, codec = codec)
				
class codec_s():

	def __init__(self, TYPE_MAX, codec = None):

		self.codec = codec
		
		codec_index = 0
		
		self.num_codec_variables          = self.codec[codec_index]; codec_index += 1
		
		self.num_base_genes               = self.codec[codec_index]; codec_index += 1
		self.num_layer_genes              = self.codec[codec_index]; codec_index += 1

		self.input_vector_num_dimensions  = self.codec[codec_index]; codec_index += 1
		self.input_vector_x_dimensions    = self.codec[codec_index]; codec_index += 1
		self.input_vector_y_dimensions    = self.codec[codec_index]; codec_index += 1
		self.input_vector_z_dimensions    = self.codec[codec_index]; codec_index += 1
		
		self.output_vector_num_dimensions = self.codec[codec_index]; codec_index += 1
		self.output_vector_x_dimensions   = self.codec[codec_index]; codec_index += 1
		self.output_vector_y_dimensions   = self.codec[codec_index]; codec_index += 1
		self.output_vector_z_dimensions   = self.codec[codec_index]; codec_index += 1
		
		self.aligment_type_limits         = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2 
		self.num_layers_limits            = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.optimizer_type_limits        = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2 
		self.loss_type_limits             = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2 
		self.learning_rate_limits         = (self.codec[codec_index]/TYPE_MAX, self.codec[codec_index + 1]/TYPE_MAX); codec_index += 2
		self.batch_size_limits            = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.num_epochs_limits      	      = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.num_semesters_limits         = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2

		self.layer_type_limits            = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.activation_present_limits    = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.activation_function_limits   = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2

		self.dense_output_x_limits        = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.dense_output_y_limits        = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.dense_output_z_limits        = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		
		self.num_kernels_limits           = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.kernel_size_x_limits         = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.kernel_size_y_limits         = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.kernel_stride_x_limits       = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.kernel_stride_y_limits       = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.kernel_dilation_y_limits     = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.kernel_dilation_x_limits     = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2

		self.pool_present_limits          = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.pool_type_limits             = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.pool_size_x_limits           = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.pool_size_y_limits           = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.pool_stride_x_limits         = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.pool_stride_y_limits         = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2

		self.batch_norm_present_limits    = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.dropout_present_limits       = (self.codec[codec_index], self.codec[codec_index + 1]); codec_index += 2
		self.dropout_value_limits         = (self.codec[codec_index]/TYPE_MAX, self.codec[codec_index + 1]/TYPE_MAX); codec_index += 2
		
		self.num_generations              = self.codec[codec_index]; codec_index += 1
		self.template_type                = self.codec[codec_index]; codec_index += 1
		
		self.checkCodecLength(codec_index)
		
		self.activation_present_limits    = self.checkBooleanBounds(self.activation_present_limits)
		self.pool_present_limits          = self.checkBooleanBounds(self.pool_present_limits)
		self.batch_norm_present_limits    = self.checkBooleanBounds(self.batch_norm_present_limits)
		self.dropout_present_limits       = self.checkBooleanBounds(self.dropout_present_limits)
		
	def checkBooleanBounds(self, limits):

		if limits[1] > 1:
			print("Warning boolean upper bound cannot be greater than 1! Assuming 1.")
			limits = (limits[0], 1)
			
		return limits
	
	def checkCodecLength(self, codec_index):
		
		if (codec_index != self.num_codec_variables):
			print(f"Error! Hardcoded python codec length \"{codec_index}\" does not equal recieved variable: \"{self.num_codec_variables}\". Exiting!")
			quit(1)

#Layer construction conditions
			
def flattenCondition(genome, layer_index, num_layers, input_size):	
	
	current_genes  = genome.layer_genes[layer_index]  

	conditions = [
		(current_genes.layer_type == 1),
		(len(input_size) != 1)
	]
	
	return all(conditions)

def layerCondition(genome, layer_index, num_layers, input_size):	
	
	conditions = [
		True
	]
	
	return all(conditions)
			
def reshapeCondition(genome, layer_index, num_layers, input_size):
	
	current_genes  = genome.layer_genes[layer_index]  

	condition_next     = True
	veto_next          = False
	
	if layer_index < (num_layers - 1):
		next_genes     = genome.layer_genes[layer_index + 1]
		condition_next = (next_genes.layer_type   != 1)
		veto_next      = (next_genes.pool_present == 1) and (next_genes.layer_type == 1)
	
	conditions = [
		(current_genes.layer_type == 1),
		condition_next
	]
	
	veto_conditions = [
		(current_genes.pool_present == 1) and (current_genes.layer_type == 1),
		(layer_index == (num_layers - 1)),
		((layer_index == (num_layers - 2)) and veto_next)
	]
		
	return all(conditions) or any(veto_conditions)

def poolCondition(genome, layer_index, num_layers, input_size):	
	
	current_genes  = genome.layer_genes[layer_index]  
	
	conditions = [
		current_genes.pool_present,
		(layer_index != (num_layers -1))
	]
	
	return all(conditions)

def batchCondition(genome, layer_index, num_layers, input_size):	
	
	current_genes  = genome.layer_genes[layer_index]  
	
	conditions = [
		current_genes.batch_norm_present,
		(layer_index != (num_layers -1))
	]
	
	return all(conditions)

def dropoutCondition(genome, layer_index, num_layers, input_size):	
	
	current_genes  = genome.layer_genes[layer_index]  
	
	conditions = [
		current_genes.dropout_present,
		(layer_index != (num_layers - 1))
	]
	
	return all(conditions)

def setupCappingLayer(codec, genes, input_size):
	
	output_shape       = popLooseDimensions([codec.output_vector_x_dimensions, codec.output_vector_y_dimensions, codec.output_vector_z_dimensions])
	genes.output_shape = tuple(output_shape)
	
	if ((genes.layer_type == 2) or (genes.layer_type == 3)):

		input_size_x = input_size[0]
		input_size_y = 1
		if (len(input_size) >= 2):
			input_size_y = input_size[1]
			
		if ((input_size_x >= codec.output_vector_x_dimensions) & (input_size_y >= codec.output_vector_y_dimensions)):

			genes.layer_type = 2

			genes.kernel_size_x = calculateConvKernelSize(input_size_x, codec.output_vector_x_dimensions, genes.kernel_stride_x, genes.kernel_dilation_x)
			genes.kernel_size_y = calculateConvKernelSize(input_size_y, codec.output_vector_y_dimensions, genes.kernel_stride_y, genes.kernel_dilation_y)

			genes.num_kernels  = codec.output_vector_z_dimensions
			
		elif ((input_size_x <= codec.output_vector_x_dimensions) & (input_size_y <= codec.output_vector_y_dimensions)):

			genes.layer_type = 3

			genes.kernel_size_x = calculateConvTKernelSize(input_size_x, codec.output_vector_x_dimensions, genes.kernel_stride_x, genes.kernel_dilation_x)
			genes.kernel_size_y = calculateConvTKernelSize(input_size_y, codec.output_vector_y_dimensions, genes.kernel_stride_y, genes.kernel_dilation_y)		

			genes.num_kernels  = codec.output_vector_z_dimensions

		else:
			genes.layer_type = 1

	if (genes.layer_type == 1):

		genes.dense_output_x = 1
		genes.dense_output_y = 1
		genes.dense_output_z = 1

		genes.dense_output_x   = output_shape[0]
		if(len(output_shape) >= 2):
			genes.dense_output_y = output_shape[1]
		if(len(output_shape) >= 3):
			genes.dense_output_z = output_shape[2]
			
	return genes
	
class Model_C(Model):
	
	def __init__(self, genome, codec, max_trainable_parameters, input_size):
		super(Model_C, self).__init__()
		
		self.genome         = genome
		self.codec          = codec
		self.num_layers     = genome.num_layers
		
		self.add_layers     = []
		self.args           = []
		self.num_parameters = 0
		
		print(f"Template Type = {codec.template_type}")
		
		template = templateType(codec.template_type)
		
		#Fit to template:
		
		#Ordered templates:
		if (template[1] == "ordered"):

			ordered_templates = dict(
				cnn         = {0: 0, 1: 1, 2: 0, 3: 3},
				autoencoder = {0: 0, 1: 1, 2: 0, 3: 3},
			)
			order_template = ordered_templates[template[0]]
			genome = orderByTemplate(order_template, genome)
		
		for layer_index in range(self.num_layers):
			
			genes              = genome.layer_genes[layer_index]
			genes.output_shape = (genes.dense_output_x, genes.dense_output_y, genes.dense_output_z)
			
			if (layer_index == (self.num_layers - 1)):
				genes = setupCappingLayer(codec, genes, input_size)
				
				#print(f"Capping layer: {addActivation(genes.activation_function)}")
				
			layer_constructors = [ 
				{"condition": flattenCondition, "layer": addFlatten  , "arguments": {}                      },
				{"condition": layerCondition  , "layer": addLayer    , "arguments": {}                      },
				{"condition": reshapeCondition, "layer": addReshape  , "arguments": {}                      },
				{"condition": poolCondition   , "layer": addPooling  , "arguments": {}                      },
				{"condition": batchCondition  , "layer": addBatchNorm, "arguments": {}                      },
				{"condition": dropoutCondition, "layer": addDropout  , "arguments": {"training": "training"}},
			]
			
			for constructor in layer_constructors:
				if constructor["condition"](genome, layer_index, self.num_layers, input_size):
					
					layer, input_size, num_parameters = constructor["layer"](genes, input_size)
					self.num_parameters += num_parameters
					
					self.add_layers.append(layer)
					self.args.append(constructor["arguments"])	
					
					try:
						activation = layer.activation
					except:
						activation = None
					
					#print(f"Layer Type: {layer}, Output Size: {input_size}, Num Parameters: {num_parameters}, Activation: {activation}")
							
		if (self.num_parameters > max_trainable_parameters):
			self.add_layers = None
			raise Exception(f"Error! Model has more parameters {self.num_parameters} than allowed {max_trainable_parameters}. Terminating creation!")

	def call(self, x, training = False):
				
		for layer_index in range(len(self.add_layers)):

			if "training" in self.args[layer_index]:
				self.args[layer_index]["training"] = training

			x = self.add_layers[layer_index](x, **self.args[layer_index])
			

				
		return x