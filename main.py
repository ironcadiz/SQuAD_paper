# Importar dependencias para el preprocesamiento.
from nltk.tokenize import word_tokenize
import string
import re
import numpy as np
import json

"""## Embeddings

Para usar los embeddings e inyectarlos en el modelo de Keras primero se intento lo siguiente:

1. Construiremos un index de palabras a partir del vocabulario del dataset.
2. Transformamos las secuencias de palabras en secuencias de enteros mediante el index.
3. Estandarizamos el Tamaño de la secuencia usando `pad_sequences` de Keras.
4. Luego  construimos matriz de pesos a partir de Glove para inyectar a una capa `Embedding` de Keras.

Este proceso no funcionó porque los vectores de glove ocupaban demasiada memoria de la gpu (eran 50M de parametros), a cambio se optó por precomputar los vectores de cada palabra y pasar los tensores de una sequencia directamente a la red. Como el dataset crece demasiado al pasar cada palabra a un vector de 300D, esta transformación se genera de a poco mediante un objeto Keras Sequence ( se probó primero un generador pero no era compatible con el paralelismo).
"""

# Importamos dependencias para generar los embeddings
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.utils import Sequence
import threading

import json
import numpy as np
import os

# Usamos los glove vectors 300D de wikipedia 2014. Por limitación de memoria no podemos usar un corpus más grande.
glove_file = datapath(os.getcwd() +'/glove.6B.300d.txt')
tmp_file = get_tmpfile(os.getcwd() + "/test_word2vec.txt")
print("Running script")
glove2word2vec(glove_file, tmp_file)
print("Loading Glove Vectors")
embedder = KeyedVectors.load_word2vec_format(tmp_file)

# Funciones para metodo Embedding layer (se desecho)
# retorna secuencia de enteros a partir de secuencia de palabras.
def text_to_sequence(text_seq, word_index):
    result = []
    for word in text_seq:
        if word in word_index:
            result.append(word_index[word])
        else:
            word_index[word] = len(word_index)
            result.append(word_index[word])
    return np.array(result)

# Construimos index de palabras a medida que aparecen en el dataset. 
# Construimos matrices de enteros a partir del dataset.
def gen_data(dataset,word_index):
    contexts = []
    questions = []
    output = []
    for document in dataset:
        for paragraph in document['paragraphs']:
            for question in paragraph['qas']:
                # Tomamos la primera respuesta para generar el dataset final
                answer = question['answers'][0]
                # Pasamos secuencias a secuencias de enteros.
                contexts.append(text_to_sequence(paragraph['context_tokenized'],word_index))
                questions.append(text_to_sequence(question['question_tokenized'],word_index))
                #guardamos tupla de inicio y fin para el output.
                output.append((answer['answer_word_start'],answer['answer_word_end']))
    return contexts,questions,output,word_index



# Funciones para la keras sequence.

# Pasamos una secuencia de texto a un tensor de tamaño fijo.
# Quedan en 0 el padding y las palabras que no estan en GLove
def text_to_tensor(text_seq, word_vectors,sequence_length):
    result = np.zeros((sequence_length,300))
    for i,t in enumerate(text_seq):
        if t in word_vectors:
            result[i]  = word_vectors[t]
    return result

def data_counter(dataset):
    count = 0
    for document in dataset:
        for paragraph in document['paragraphs']:
            for question in paragraph['qas']:
                count +=1
    return count

# Secuencia para hacer multijob data generation
class TensorSequence(Sequence):

    def __init__(self, dataset,batch_size,word_vectors,context_length,question_length):
        self.dataset = []
        self.batch_size = batch_size
        self.data_count = data_counter(dataset)
        self.word_vectors = word_vectors
        self.context_length = context_length
        self.question_length = question_length
        print("Loading sequence")
        # Guardamos el dataset en formato tabla para poder indexar por batch size.
        for document in dataset:
            for paragraph in document['paragraphs']:
                for question in paragraph['qas']:
                    # Tomamos la primera respuesta para generar el dataset final
                    answer = question['answers'][0]
                    context = paragraph['context_tokenized'] if len(paragraph['context_tokenized']) <= context_length else paragraph['context_tokenized'][:context_length]
                    question_t = question['question_tokenized'] if len(question['question_tokenized']) <= question_length else question['question_tokenized'][:question_length]
                    self.dataset.append([context,question_t,min(answer['answer_word_start'],context_length-1),min(answer['answer_word_end'],context_length-1)])


    # steps per batch
    def __len__(self):
        return self.data_count//self.batch_size

    
    
    # retorna un batch de tensores.
    def __getitem__(self, idx):
        contexts = None
        questions = None
        output_start = []
        output_end = []
        #iteramos sobre el batch pedido.
        for row in self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]:
            
            # pasamos sequencias de palabras a tensores
            if contexts is None:
                contexts = text_to_tensor(row[0],self.word_vectors,self.context_length)
            else:
                contexts = np.dstack((contexts,text_to_tensor(row[0],self.word_vectors,self.context_length)))
            if  questions is None:
                questions = text_to_tensor(row[1],self.word_vectors,self.question_length)
            else:
                questions= np.dstack((questions,text_to_tensor(row[1],self.word_vectors,self.question_length)))

            #guardamos tupla de inicio y fin para el output.
            output_start.append(row[2])
            output_end.append(row[3])
        return [np.moveaxis(contexts,2,0),np.moveaxis(questions,2,0)],[to_categorical(np.array(output_start),num_classes=self.context_length),to_categorical(np.array(output_end),num_classes=self.context_length)]

# generamos las secuencias de enteros para inyectar en el modelo de Keras.
with open("train-v1.1-pr.json", "r") as data:
    train = json.load(data)
with open("dev-v1.1-pr.json", "r") as data:
    test = json.load(data)

# Calculamos el tamaño máximo
def max_context(document):
    return max(document['paragraphs'], key= lambda x: len(x['context_tokenized']))

def max_question_par(paragraph):
    return max(paragraph['qas'], key= lambda x: len(x['question_tokenized']))

def max_paragraph(document):
    return max(document['paragraphs'], key=lambda x: len(max_question_par(x)['question_tokenized']))

TRAIN_COUNT = data_counter(train)
MAX_CONTEXT = 400
MAX_QUESTIONS = 30

"""## Declaración del Modelo"""

# Importamos dependencias
from keras.layers import Input, Concatenate, Dense, Reshape, Activation,Multiply, Dot, Add, Lambda,SeparableConv1D, BatchNormalization,TimeDistributed,Dropout,Reshape,Softmax, Reshape, Flatten
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf
from keras import backend as K
import keras.backend as K
import keras.initializers
import numpy as np
# Para elegir GPU o multicore
'''
num_cores = 4
CPU= False
GPU= not CPU
if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)
'''
## Attention 

class ScaledDotProductAttention():
	def __init__(self, d_model, attn_dropout=0.1):
		self.temper = np.sqrt(d_model)
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k, v, mask):
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
		if mask is not None:
			mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
			attn = Add()([attn, mmask])
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output

#https://github.com/Lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py
class MultiHeadAttention():
	# mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout):
      self.n_head = n_head
      self.d_k = d_k
      self.d_v = d_v
      self.dropout = dropout
      self.qs_layers = []
      self.ks_layers = []
      self.vs_layers = []
      for _ in range(n_head):
        self.qs_layers.append(TimeDistributed(Dense(d_k, kernel_regularizer=l2(3e-7),use_bias=False)))
        self.ks_layers.append(TimeDistributed(Dense(d_k, kernel_regularizer=l2(3e-7),use_bias=False)))
        self.vs_layers.append(TimeDistributed(Dense(d_v, kernel_regularizer=l2(3e-7),use_bias=False)))
      self.attention = ScaledDotProductAttention(d_model)
      #self.layer_norm = BatchNormalization(axis=1)
      self.w_o = TimeDistributed(Dense(d_model,kernel_regularizer=l2(3e-7), bias_regularizer=l2(3e-7)))

    def __call__(self, q, k, v, mask=None):
      d_k, d_v = self.d_k, self.d_v
      n_head = self.n_head
      heads = []
      #attns = []
      for i in range(n_head):
        qs = self.qs_layers[i](q)   
        ks = self.ks_layers[i](k) 
        vs = self.vs_layers[i](v) 
        #head, attn = self.attention(qs, ks, vs, mask)
        head = self.attention(qs, ks, vs, mask)
        heads.append(head)
        #attns.append(attn)
      head = Concatenate()(heads)
      #attn = Concatenate()(attns)

      outputs = self.w_o(head)
      outputs = Dropout(self.dropout)(outputs)
      outputs = Add()([outputs, q])
      return outputs
    
    
class EncoderConv():
  
    def __init__(self,n_convs,filters,kernel,name="encoder_conv"):
        self.n_convs = n_convs
        self.filters = filters
        self.kernel = kernel
        self.name = name
        self.norms = []
        self.convs = []
        for i in range(self.n_convs):
            norm_layer = BatchNormalization(axis = 1,name=name+"_norm_{}".format(i))
            conv_layer = SeparableConv1D(filters=filters,kernel_size=kernel,depthwise_regularizer=l2(3e-7),pointwise_regularizer=l2(3e-7),bias_regularizer=l2(3e-7),name=name+"_conv_{}".format(i),padding="same")
            self.norms.append(norm_layer)
            self.convs.append(conv_layer)


    def __call__(self,value):
        value_normed = self.norms[0](value)
        value_conv = self.convs[0](value_normed)
        value_end = value_conv
        for i in range(1,self.n_convs):
          value_normed = value_normed = self.norms[i](value_end)
          value_conv = self.convs[i](value_normed)
          value_end = Add()([value_conv,value_end])

        return value_end

class SelfAttention():
  
    def __init__(self,n_heads,d_model,d_k,d_v,dropout=0.1,name="encoder_self_attention"):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.name = name
        self.attn = MultiHeadAttention(n_heads,d_model,d_k,d_v,dropout)
        self.norm_layer = BatchNormalization(axis = 1,name=name+"_norm")
        #self.mask = Lambda(lambda x:GetPadMask(x,x))(norm_layer)
        
    def __call__(self,value):
        norm_layer = self.norm_layer(value)
        attn_layer = self.attn(norm_layer,norm_layer,norm_layer)
        value = Add()([value,attn_layer])
        return value
      
class FeedForward():
  
    def __init__(self,ndims,activation="relu",name="encoder_ff"):
        self.ndims = ndims
        self.activation = activation
        self.name = name
        self.norm_layer = BatchNormalization(axis=1,name=name+"_norm")
        self.ff = Dense(ndims, kernel_regularizer=l2(3e-7),bias_regularizer=l2(3e-7),activation=activation, name=name+"_ff")
      
    def __call__(self,value):
        norm = self.norm_layer(value)
        ff = self.ff(norm)
        value = Add()([value,ff])
        return value

      
class EncoderBlock():
  
  ''' 
      Ensamble de Encoder Block
      Para las Stacked Embedding EB, n_conv = 4
      Para las Stacked Model EB, n_conv = 2 (Luego necesito repetir el EB 7 veces y tener 3 repeticiones de eso con pesos compartidos)
  
  '''
  
  def __init__(self, n_convs, filters, kernels, n_heads, d_model, d_k, d_v, ndims, dropout=0.1, activation="relu", name="encoder_block"):
    self.name = name
    self.dropout = dropout
    self.encoder_conv = EncoderConv(n_convs, filters, kernels, name=name+"_conv")
    self.self_attention = SelfAttention(n_heads, d_model, d_k, d_v, dropout,name=name+"_self_attention")
    self.ff = FeedForward(ndims, activation,name=name+"_ff")
    
  def __call__(self, value):
    enc_conv = self.encoder_conv(value)
    self_att = self.self_attention(enc_conv)
    value = self.ff(self_att)
    return value
  
class ModelEncoder():
  
  '''
     Concatenación de n_reps Enconder blocks
  '''
  
  def __init__(self, n_reps, n_convs, filters, kernels, n_heads, d_model, d_k, d_v, ndims, dropout=0.1, activation="relu", name="model_encoder"):
    self.blocks = []
    self.n_reps = n_reps
    self.name = name
    for i in range(self.n_reps):
      self.blocks.append(EncoderBlock(n_convs, filters, kernels, n_heads, d_model, d_k, d_v, ndims, dropout,name=name+"_block_{}".format(i)))

  def __call__(self, value):
    for i in range(self.n_reps):
      value = self.blocks[i](value)
    return value

## Highway network https://gist.github.com/iskandr/a874e4cf358697037d14a17020304535
def highway_layers(value, n_layers, activation="tanh", gate_bias=-3,name="highway"):
    dim = K.int_shape(value)[-1]
    gate_bias_initializer = keras.initializers.Constant(gate_bias)
    for i in range(n_layers):     
        gate = Dense(units=dim, bias_initializer=gate_bias_initializer,kernel_regularizer=l2(3e-7),bias_regularizer=l2(3e-7),name=name+"_dense_1_{}".format(i))(value)
        gate = Activation("sigmoid",name=name+"_activation_1_{}".format(i))(gate)
        negated_gate = Lambda(
            lambda x: 1.0 - x,
            output_shape=(dim,))(gate)
        transformed = Dense(units=dim,kernel_regularizer=l2(3e-7),bias_regularizer=l2(3e-7),name=name+"_dense_2_{}".format(i))(value)
        transformed = Activation(activation,name=name+"_activation_2_{}".format(i))(value)
        transformed_gated = Multiply(name=name+"_multiply_1_{}".format(i))([gate, transformed])
        identity_gated = Multiply(name=name+"_multiply_2_{}".format(i))([negated_gate, value])
        value = Add(name=name+"_add_{}".format(i))([transformed_gated, identity_gated])
    return value
  
def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask
  
def create_mask(x):
    zeros = K.zeros_like(x)
    return K.cast(K.not_equal(zeros,x), dtype='float32')

def attention(batch):
  
  def _attention_f(c_q):
      c,q=c_q[:MAX_CONTEXT,:], c_q[MAX_CONTEXT:,:]
      c = K.tile(c,[MAX_QUESTIONS,1])
      q = K.reshape(K.tile(q,[1,MAX_CONTEXT]),[MAX_QUESTIONS*MAX_CONTEXT,FILTERS])
      return K.concatenate([q,c,c*q],axis=1)
    
  return K.map_fn(_attention_f,batch)

## Model params
GLOVE_DIM=300
KERNEL_SIZE=7
FILTERS=64
BLOCK_CONV_LAYERS=4
N_HEADS=4
DROPOUT=0.1
N_REPS = 3
BLOCK_CONV_LAYERS_STACKED = 2
STACKED_KERNEL_SIZE=5

## Question embedding
question_input = Input(shape=(MAX_QUESTIONS,GLOVE_DIM),name="question_input")
highway_question = highway_layers(question_input,2,activation="relu", gate_bias=-3,name="question_highway")
question_ff = EncoderBlock(BLOCK_CONV_LAYERS,FILTERS,KERNEL_SIZE,N_HEADS,FILTERS,FILTERS,FILTERS,FILTERS,DROPOUT,name="question_eeb")(highway_question)

## context embedding
context_input = Input(shape=(MAX_CONTEXT,GLOVE_DIM),name="context_input")
highway_context = highway_layers(context_input,2,activation="relu", gate_bias=-3,name="context_highway")
context_ff = EncoderBlock(BLOCK_CONV_LAYERS,FILTERS,KERNEL_SIZE,N_HEADS,FILTERS,FILTERS,FILTERS,FILTERS,DROPOUT,name="context_eeb")(highway_context)

## Context question attention
concat = Concatenate(axis=1)([context_ff,question_ff])
lambda_concat = Lambda(attention)(concat)
attention_dense = TimeDistributed(Dense(1,kernel_regularizer=l2(3e-7),use_bias=False))(lambda_concat)
attention_matrix = Reshape((MAX_CONTEXT,MAX_QUESTIONS))(attention_dense)
attention_matrix_bar = Softmax()(attention_matrix)
A = Dot(axes=(2,1))([attention_matrix_bar, question_ff])

attention_matrix_transpose = Lambda(lambda x : K.permute_dimensions(x, (0, 2, 1)))(attention_matrix)
attention_matrix_bar_bar = Softmax()(attention_matrix_transpose)
B = Dot(axes=(2,1))([attention_matrix_bar, attention_matrix_bar_bar])
B = Dot(axes=(2,1))([B, context_ff])

## Stacked model encoder blocks.
A_attention = Multiply()([context_ff,A])
B_attention = Multiply()([context_ff,B])

stacked_blocks_input=Concatenate(axis=2)([context_ff,A,A_attention,B_attention])

stacked_blocks_resized = SeparableConv1D(filters=FILTERS,kernel_size=STACKED_KERNEL_SIZE,depthwise_regularizer=l2(3e-7),pointwise_regularizer=l2(3e-7),bias_regularizer=l2(3e-7),name="conv_resize",padding="same")(stacked_blocks_input)


me = ModelEncoder(N_REPS, BLOCK_CONV_LAYERS_STACKED,FILTERS,STACKED_KERNEL_SIZE,N_HEADS,FILTERS,FILTERS,FILTERS,FILTERS,DROPOUT)

stacked_encoder_blocks_0 = me(stacked_blocks_resized)
stacked_encoder_blocks_1 = me(stacked_encoder_blocks_0)
stacked_encoder_blocks_2 = me(stacked_encoder_blocks_1)

## Output layer

start_layer = Concatenate(axis=2)([stacked_encoder_blocks_0,stacked_encoder_blocks_1]) # no estoy seguro del axis
start_dense = TimeDistributed(Dense(1,kernel_regularizer=l2(3e-7),use_bias=False))(start_layer)
start_reshape = Flatten()(start_dense)
start_output = Softmax()(start_reshape)


end_layer = Concatenate(axis=2)([stacked_encoder_blocks_0,stacked_encoder_blocks_2]) # no estoy seguro del axis
end_dense = TimeDistributed(Dense(1,kernel_regularizer=l2(3e-7), use_bias=False))(end_layer)
end_reshape = Flatten()(end_dense)
end_output = Softmax()(end_reshape)

model = Model(inputs=[context_input,question_input] ,outputs =[start_output,end_output])
model.summary()

"""## Entrenamiento

Se entrenó en una máquina con 8Gb de RAM, 4 Cores de CPU y una GTX1050M de 4Gb.

Se utilizo `Adam` con `batch_size` de `32`, `learning_rate` `0.001` y categorical cross entropy como función de pérdida, sumando las pérdidas de la palabra inicial y la palabra final de la respuesta.

Al entrenar no se paso un set de validación puesto que la memoria era un limitante fuerte y simplemente no había espacio. Es más se redujo el tamaño de la cola de batches a 5.

Finalmente aprovechando que la generación de batches se puede paralelizar mediante una `keras.utils.Sequence`, se usan 3 workers para alimentar el entrenamiento.

Luego de mucho optimizar e iterar se logró bajar el tiempo de entrenamiento a 3.5 horas por época,  lo que se mantuvo por alrededor de 16 horas llegando a 5 épocas, con un training accuracy de 4% y 2% para las palabras inicial y final.

Luego se paro el entrenamiento y se volvió a iniciar esta véz con un learning rate mas alto (`0.003`), lo cual funcionó bien inicialmente hasta que en algun momento el gradiente explotó y el accuracy bajo estrepitosamente.
"""

class History(Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}
        self.epochs_count = 0
        self.metrics_path = 'metrics.txt'
        with open(self.metrics_path, 'a') as fp:
            fp.write('epoch\t softmax_3_acc\t softmax_4_acc\t softmax_3_loss\t softmax_4_loss\t val_softmax_3_acc\t val_softmax_4_acc\t val_softmax_3_loss\t val_softmax_4_loss\t \n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        self.epochs_count += 1
        print(logs)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
          
        with open(self.metrics_path, 'a') as fp:
            fp.write('{}\t {}\t {}\t {}\t {}\t {}\t {}\t {}\t {}\t \n'.format(self.epochs_count,
                                                                              self.history['softmax_3_acc'][-1], 
                                                                              self.history['softmax_4_acc'][-1], 
                                                                              self.history['softmax_3_loss'][-1], 
                                                                              self.history['softmax_4_loss'][-1],
                                                                              self.history['val_softmax_3_acc'][-1],
                                                                              self.history['val_softmax_4_acc'][-1],
                                                                              self.history['val_softmax_3_loss'][-1],
                                                                              self.history['val_softmax_4_loss'][-1],
                                                                             
                                                                             ))



# Parametros
BATCH_SIZE=16
EPOCHS=50
OPTIMIZER=Adam(beta_1=0.8, beta_2=0.999, epsilon=1e-7)
LOSS= 'categorical_crossentropy'
generator= TensorSequence(train, BATCH_SIZE, embedder, MAX_CONTEXT, MAX_QUESTIONS)
dev_generator = TensorSequence(test, BATCH_SIZE, embedder, MAX_CONTEXT, MAX_QUESTIONS)
checkpoint = ModelCheckpoint(filepath='weights.hdf5',monitor="loss", verbose=1,save_weights_only=True)
history = History()

callbacks_list = [checkpoint, history]

model.compile(optimizer=OPTIMIZER,loss=LOSS, metrics=['accuracy'])
model.load_weights('weights.hdf5')
model.fit_generator(generator, validation_data=dev_generator, steps_per_epoch = TRAIN_COUNT//BATCH_SIZE, max_queue_size=5, epochs = EPOCHS, verbose=1, callbacks=callbacks_list, use_multiprocessing=True, workers=3)

#entrenamos desde archivo guardado
'''
model = load_model('weights.hdf5')
BATCH_SIZE=32
EPOCHS=50
OPTIMIZER= Adam(lr=0.003) #nuevo learning rate
LOSS= 'categorical_crossentropy'
generator= TensorSequence(train,BATCH_SIZE,embedder,MAX_CONTEXT,MAX_QUESTIONS)

checkpoint = ModelCheckpoint(filepath='weights.hdf5',monitor="loss", verbose=1)
callbacks_list = [checkpoint]

model.summary()

model.compile(optimizer=OPTIMIZER,loss=LOSS, metrics=['accuracy'])
model.fit_generator(generator, steps_per_epoch = TRAIN_COUNT//BATCH_SIZE, max_queue_size=5, epochs = EPOCHS, verbose=1, callbacks=callbacks_list, use_multiprocessing=True, workers=3)
'''
