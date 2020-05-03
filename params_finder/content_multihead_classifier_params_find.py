# coding='UTF-8'
import io
import sys
import os
import re
import pandas as pd
import nltk
import nltk.data
import numpy as np
import tensorflow
import keras
from keras import regularizers
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.engine.topology import Layer
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Bidirectional, Flatten, Dropout, Multiply, Permute, concatenate
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold

from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

from evaluation_metrics import  precision, recall, fbeta_score, fmeasure, getAccuracy

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')  #解决print输出报编码错问题

def prepare_input(file_num_limit, paras_limit):
    sampled_para_encoded_dir = './encoded_para_sep/'  # encoded_para_sep_sampled
    sampled_para_txt_list = './labels_feature_stats_merged_cleaned20200223.csv'  # plain_txt_name_index_sampled.csv

    para_encoded_txts = os.listdir(sampled_para_encoded_dir)
    sampled_label_data = pd.read_csv(sampled_para_txt_list, encoding='utf-8-sig')

    # 取得全部文章的encoded content, 并将其解析成为(paras_limit, 768) 的shape
    contents = np.zeros((file_num_limit, paras_limit, 768))
    labels = []
    stats_features = []
    file_number_count = 0
    for file_index in range(len(para_encoded_txts)):
        file = para_encoded_txts[file_index]
        # 调整X的输入
        content = np.loadtxt(sampled_para_encoded_dir + file)
        content = content.reshape(-1, 768)
        # 调整Y的输入
        label_index = sampled_label_data[(sampled_label_data['file_name']==int(file.replace('.txt', '')))].index
        label = np.array(sampled_label_data.iloc[label_index, 2:8]).tolist()
        if len(label) != 0:
            # 截断超过paras_limit个元素的content （para中指section超过paras_limit个的文章）
            for para_index in range(min(paras_limit, content.shape[0])):
                contents[file_number_count, para_index, :] = content[para_index, :]
            stats_feature = np.array(sampled_label_data.iloc[label_index, 10:]).tolist()
            # print(label_index, sampled_label_data['article_names'][label_index], label)
            labels.append(label)
            stats_features.append(stats_feature)
            file_number_count += 1

        if file_number_count + 1 >= file_num_limit:
            break
   
    # 仅存储有值部分的文本，去除后面的空值
    contents = contents[:file_number_count,:,:]
    
    # 将Y转化为numpy表达，把格式调整成： n*1*6 -> n*6
    onehotlabels = [label[0] for label in labels]
    stats_features = [stats_feature[0] for stats_feature in stats_features]

    return contents, onehotlabels, stats_features


# target params
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                     name='learning_rate')
dim_hidden_head = Integer(low=16, high=64, name='hidden_head')
dim_multiheads = Integer(low=16, high=64, name='multiheads')
dim_epochs = Integer(low=10, high=30, name='epochs')
dim_batch_size = Integer(low=16, high=256, name='batch_size')
dim_adam_decay = Real(low=1e-6,high=1e-2, name="adam_decay")

dimensions = [dim_learning_rate,
              dim_hidden_head,
              dim_multiheads,
              dim_epochs,
              dim_batch_size,
              dim_adam_decay
             ]
default_parameters = [1e-4, 32, 32, 20, 100, 1e-6]

# input prepare
# 输入限制，文章数量及每篇文章输入的句子/段落数
file_num_limit = 45614  # total 45614
paras_limit = 20

encoded_contents, onehotlabels, stats_features = prepare_input(file_num_limit, paras_limit)

# stats_features 标准化
scaler = preprocessing.StandardScaler() #实例化
scaler = scaler.fit(stats_features)
stats_features = scaler.transform(stats_features)

# 换算成二分类
# no_footnotes-0, primary_sources-1, refimprove-2, original_research-3, advert-4, notability-5
no_good_flaw_type = 4 # finished
# 找出FA类的索引
FA_indexs = [index for index in range(len(onehotlabels)) if sum([int(item) for item in onehotlabels[index]]) == 0]
# 找出二分类另外一类的索引
not_good_indexs = [index for index in range(len(onehotlabels)) if onehotlabels[index][no_good_flaw_type] > 0]
binary_classification_indexs = FA_indexs + not_good_indexs
print('FA count:', len(FA_indexs), 'no good count:', len(not_good_indexs))
encoded_contents = np.array([encoded_contents[index,:,:] for index in binary_classification_indexs])
onehotlabels = np.array([onehotlabels[index] for index in binary_classification_indexs])
stats_features = np.array([stats_features[index] for index in binary_classification_indexs])

# 变成二分类的标签
onehotlabels = onehotlabels[:,no_good_flaw_type]
onehotlabels = np.array([[label] for label in onehotlabels])
### split data into training set and label set
# X_train, X_test, y_train, y_test = train_test_split(encoded_contents, onehotlabels, test_size=0.1, random_state=42)
X_train_content, X_train_stats, y_train = encoded_contents, stats_features, onehotlabels

# 打乱操作
index = [i for i in range(len(y_train))]
np.random.shuffle(index)

X_train_content = encoded_contents[index]
X_train_stats = stats_features[index]
y_train = onehotlabels[index]

X_train = [X_train_content, X_train_stats]
y_train = y_train


# model
class Position_Embedding(Layer):
    
    def __init__(self,method='sum',embedding_dim=None,**kwargs):
        """
        # 此层Layer仅可放在Embedding之后。
        # 参数：
        #    - embedding_dim: position_embedding的维度，为None或者偶数（Google给的Position_Embedding构造公式分奇偶数）；
        #    - method: word_embedding与position_embedding的结合方法，求和sum或拼接concatenate；
        #        -- sum: position_embedding的值与word_embedding相加，需要将embedding_dim定义得和word_embedding一样；默认方式，FaceBook的论文和Google论文中用的都是后者；
        #        -- concatenate：将position_embedding的值拼接在word_embedding后面。
        """
        self.method = method
        self.embedding_dim = embedding_dim
        super(Position_Embedding,self).__init__(**kwargs)
        
    def compute_output_shape(self,input_shape):
        if self.method == 'sum':
            return input_shape
        elif self.method == 'concatenate':
            return (input_shape[0],input_shape[1],input_shape[2]+self.embedding_dim)
        else:
            raise TypeError('Method not understood:', self.method)
    
    def call(self,word_embeddings):
        """
        # 参照keras.engine.base_layer的call方法。
        # 将word_embeddings中的第p个词语映射为一个d_pos维的position_embedding，其中第i个元素的值为PE_i(p)，计算公式如下：
        #     PE_2i(p) = sin(p/10000^(2i/d_pos))
        #     PE_2i+1(p) = cos(p/10000^(2i/d_pos))
        # 参数
        #     - word_embeddings: Tensor or list/tuple of tensors.
        # 返回
        #     - position_embeddings：Tensor or list/tuple of tensors.
        """
        if (self.embedding_dim == None) or (self.method == 'sum'):
            self.embedding_dim = int(word_embeddings.shape[-1])
        batch_size,sequence_length = K.shape(word_embeddings)[0],K.shape(word_embeddings)[1]
        # 生成(self.embedding_dim,)向量：1/(10000^(2*[0,1,2,...,self.embedding_dim-1]/self.embedding_dim))，对应公式中的1/10000^(2i/d_pos)
        embedding_wise_pos = 1. / K.pow(10000.,2*K.arange(self.embedding_dim/2,dtype='float32')/self.embedding_dim) #n_dims=1, shape=(self.embedding_dim,)
        # 增加维度
        embedding_wise_pos = K.expand_dims(embedding_wise_pos,0) #n_dims=2, shape=(1,self.embedding_dim)
        # 生成(batch_size,sequence_length,)向量，基础值为1，首层为0，按层累加(第一层值为0，第二层值为1，...)，对应公式中的p
        word_wise_pos = K.cumsum(K.ones_like(word_embeddings[:,:,0]),axis=1) - 1 #n_dims=2, shape=(batch_size,sequence_length)
        # 增加维度
        word_wise_pos = K.expand_dims(word_wise_pos,2) #n_dims=3, shape=(batch_size,sequence_length,1)
        # 生成(batch_size,sequence_length,self.embedding_dim)向量，对应公式中的p/10000^(2i/d_pos)
        position_embeddings = K.dot(word_wise_pos,embedding_wise_pos)
        #直接concatenate无法出现交替现象，应先升维再concatenate再reshape
        position_embeddings = K.reshape(K.concatenate([K.cos(position_embeddings),K.sin(position_embeddings)],axis=-1),shape=(batch_size,sequence_length,-1))
        if self.method == 'sum':
            return word_embeddings + position_embeddings
        elif self.method == 'concatenate':
            return K.concatenate([word_embeddings,position_embeddings],axis=-1)

class Attention(Layer):
    
    def __init__(self,multiheads,head_dim,mask_right=False,**kwargs):
        """
        # 参数：
        #    - multiheads: Attention的数目
        #    - head_dim: Attention Score的维度
        #    - mask_right: Position-wise Mask，在Encoder时不使用，在Decoder时使用
        """
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        super(Attention, self).__init__(**kwargs)
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0][0],input_shape[0][1],self.output_dim) #shape=[batch_size,Q_sequence_length,self.multiheads*self.head_dim]

    def build(self,input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),#input_shape[0] -> Q_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),#input_shape[1] -> K_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),#input_shape[2] -> V_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
    
    def Mask(self,inputs,seq_len,mode='add'):
        """
        # 需要对sequence进行Mask以忽略填充部分的影响，一般将填充部分设置为0。
        # 由于Attention中的Mask要放在softmax之前，则需要给softmax层输入一个非常大的负整数，以接近0。
        # 参数：
        #    - inputs: 输入待mask的sequence
        #    - seq_len: shape=[batch_size,1]或[batch_size,]
        #    - mode: mask的方式，'mul'时返回的mask位置为0，'add'时返回的mask位置为一个非常大的负数，在softmax下为0。由于attention的mask是在softmax之前，所以要用这种方式执行
        """
        if seq_len == None:
            return inputs
        else:
            #seq_len[:,0].shape=[batch_size,1]
            #short_sequence_length=K.shape(inputs)[1]：较短的sequence_length，如K_sequence_length，V_sequence_length
            mask = K.one_hot(indices=seq_len[:,0],num_classes=K.shape(inputs)[1])#mask.shape=[batch_size,short_sequence_length],mask=[[0,0,0,0,1,0,0,..],[0,1,0,0,0,0,0...]...]
            mask = 1 - K.cumsum(mask,axis=1)#mask.shape=[batch_size,short_sequence_length],mask=[[1,1,1,1,0,0,0,...],[1,0,0,0,0,0,0,...]...]
            #将mask增加到和inputs一样的维度，目前仅有两维[0],[1]，需要在[2]上增加维度
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            #mask.shape=[batch_size,short_sequence_length,1,1]
            if mode == 'mul':
                #Element-wise multiply：直接做按位与操作
                #return_shape = inputs.shape
                #返回值：[[seq_element_1,seq_element_2,...,masked_1,masked_2,...],...]，其中seq_element_i,masked_i的维度均为2维
                #masked_i的值为0
                return inputs * mask
            elif mode == 'add':
                #Element-wise add：直接做按位加操作
                #return_shape = inputs.shape
                #返回值：[[seq_element_1,seq_element_2,...,masked_1,masked_2,...],...]，其中seq_element_i,masked_i的维度均为2维
                #masked_i的值为一个非常大的负数，在softmax下为0。由于attention的mask是在softmax之前，所以要用这种方式执行
                return inputs - (1 - mask) * 1e12
    
    def call(self,QKVs):
        """
        # 参照keras.engine.base_layer的call方法。
        # 1. Q',K',V' = Q .* WQ_i,K .* WK_i,V .* WV_i
        # 2. head_i = Attention(Q',K',V') = softmax((Q' .* K'.T)/sqrt(d_k)) .* V
        # 3. MultiHead(Q,K,V) = Concat(head_1,...,head_n)
        # 参数
            - QKVs：[Q_seq,K_seq,V_seq]或[Q_seq,K_seq,V_seq,Q_len,V_len]
                -- Q_seq.shape = [batch_size,Q_sequence_length,Q_embedding_dim]
                -- K_seq.shape = [batch_size,K_sequence_length,K_embedding_dim]
                -- V_seq.shape = [batch_size,V_sequence_length,V_embedding_dim]
                -- Q_len.shape = [batch_size,1],如：[[7],[5],[3],...]
                -- V_len.shape = [batch_size,1],如：[[7],[5],[3],...]
        # 返回
            - 
        """
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(QKVs) == 3:
            Q_seq,K_seq,V_seq = QKVs
            Q_len,V_len = None,None
        elif len(QKVs) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = QKVs
        #对Q、K、V做线性变换，以Q为例进行说明
        #Q_seq.shape=[batch_size,Q_sequence_length,Q_embedding_dim]
        #self.WQ.shape=[Q_embedding_dim,self.output_dim]=[Q_embedding_dim,self.multiheads*self.head_dim] 
        Q_seq = K.dot(Q_seq,self.WQ)#Q_seq.shape=[batch_size,Q_sequence_length,self.output_dim]=[batch_size,Q_sequence_length,self.multiheads*self.head_dim] 
        Q_seq = K.reshape(Q_seq,shape=(-1,K.shape(Q_seq)[1],self.multiheads,self.head_dim))#Q_seq.shape=[batch_size,Q_sequence_length,self.multiheads,self.head_dim]
        Q_seq = K.permute_dimensions(Q_seq,pattern=(0,2,1,3))#Q_seq.shape=[batch_size,self.multiheads,Q_sequence_length,self.head_dim]
        #对K做线性变换，和Q一样
        K_seq = K.dot(K_seq,self.WK)
        K_seq = K.reshape(K_seq,shape=(-1,K.shape(K_seq)[1],self.multiheads,self.head_dim))
        K_seq = K.permute_dimensions(K_seq,pattern=(0,2,1,3))
        #对V做线性变换，和Q一样
        V_seq = K.dot(V_seq,self.WV)
        V_seq = K.reshape(V_seq,shape=(-1,K.shape(V_seq)[1],self.multiheads,self.head_dim))
        V_seq = K.permute_dimensions(V_seq,pattern=(0,2,1,3))
        #计算内积
        A = K.batch_dot(Q_seq,K_seq,axes=[3,3])/K.sqrt(K.cast(self.head_dim,dtype='float32'))#A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        A = K.permute_dimensions(A,pattern=(0,3,2,1))#A.shape=[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]
        #Mask部分：
        #1.Sequence-wise Mask(axis=1)：这部分不是Attention论文提出的操作，而是常规应该有的mask操作（类似于Keras.pad_sequence）
        #原始输入A的形状，[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]
        #这部分是为了mask掉sequence的填充部分，比如V_len=5,那么对于A需要在K_sequence_length部分进行mask
        #这部分不好理解的话可以想象为在句子长度上进行mask，统一对齐到V_len
        A = self.Mask(A,V_len,'add')
        A = K.permute_dimensions(A,pattern=(0,3,2,1))#A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        #2.Position-wise Mask(axis=2)：这部分是Attention论文提出的操作，在Encoder时不使用，在Decoder时使用
        #原始输入A的形状，[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        #这部分是为了mask掉后续Position的影响，确保Position_i的预测输出仅受Position_0~Position_i的影响
        #这部分不好理解的话可以想象为为进行实时机器翻译时，机器是无法获取到人后面要说的是什么话，它能获得的信息只能是出现过的词语
        if self.mask_right:
            ones = K.ones_like(A[:1,:1]) #ones.shape=[1,1,Q_sequence_length,K_sequence_length],生成全1矩阵
            lower_triangular = K.tf.matrix_band_part(ones,num_lower=-1,num_upper=0) #lower_triangular.shape=ones.shape，生成下三角阵
            mask = (ones - lower_triangular) * 1e12 #mask.shape=ones.shape，生成类上三角阵（注：这里不能用K.tf.matrix_band_part直接生成上三角阵，因为对角线元素需要丢弃），同样需要乘以一个很大的数（减去这个数）,以便在softmax时趋于0
            A = A - mask #Element-wise subtract，A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        A = K.softmax(A) #A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        #V_seq.shape=[batch_size,V_sequence_length,V_embedding_dim]
        O_seq = K.batch_dot(A,V_seq,axes=[3,2])#O_seq.shape=[batch_size,self.multiheads,Q_sequence_length,V_sequence_length]
        O_seq = K.permute_dimensions(O_seq,pattern=(0,2,1,3))#O_seq.shape=[batch_size,Q_sequence_length,self.multiheads,V_sequence_length]
        #这里有个坑，维度计算时要注意：(batch_size*V_sequence_length)/self.head_dim要为整数
        O_seq = K.reshape(O_seq,shape=(-1,K.shape(O_seq)[1],self.output_dim))#O_seq.shape=[,Q_sequence_length,self.multiheads*self.head_dim]
        O_seq = self.Mask(O_seq,Q_len,'mul')
        return O_seq

# multi head attention loss function
def binary_crossentropy_add_mi(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) + 1e-9


def Multi_Head_Attention_sematic(X_train_content, y_train, learning_rate, hidden_head, multiheads, adam_decay):
    # content part
    content_input = Input(shape=(X_train_content.shape[1],X_train_content.shape[2]), name='content_bert_input')
    embedding = Position_Embedding()(content_input)
    attention = Attention(multiheads=multiheads,head_dim=hidden_head,mask_right=False)([embedding,embedding,embedding])
    # attention_layer_norm = LayerNormalization()(attention)
    # 添加attention层
    x = Flatten()(attention)
    content_feedforward_1 = Dense(256, activation='relu', name='main_feedforward_1')(x)
    content_feedforward_2 = Dense(128, activation='relu', name='main_feedforward_2')(content_feedforward_1)
    content_feedforward_3 = Dense(64, activation='relu', name='main_feedforward_3')(content_feedforward_2)
    
    possibility_outputs = Dense(1, activation='sigmoid', name='label_output')(content_feedforward_3)  # softmax  sigmoid
    
    adam = Adam(lr=learning_rate, decay= adam_decay)  # , clipnorm=0.5
    model = Model(inputs=content_input, outputs=possibility_outputs)  # stats_input
    model.compile(loss=binary_crossentropy_add_mi, optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    return model


def Multi_Head_Attention_sematic_stats(X_train, y_train, learning_rate, hidden_head, multiheads, adam_decay):
    # content part
    X_train_content = X_train[0]
    X_train_stats = X_train[1]
    content_input = Input(shape=(X_train_content.shape[1],X_train_content.shape[2]), name='content_bert_input')
    embedding = Position_Embedding()(content_input)
    attention = Attention(multiheads=multiheads,head_dim=hidden_head,mask_right=False)([embedding,embedding,embedding])
    # attention_layer_norm = LayerNormalization()(attention)
    # 添加attention层
    x = Flatten()(attention)
    content_feedforward_1 = Dense(256, activation='relu', name='main_feedforward_1')(x)
    content_feedforward_2 = Dense(128, activation='relu', name='main_feedforward_2')(content_feedforward_1)
    content_feedforward_3 = Dense(64, activation='relu', name='main_feedforward_3')(content_feedforward_2)

    # stats part
    stats_input = Input(shape=(65,), name='stats_input')
    concat_layer = concatenate([content_feedforward_3, stats_input])
    x = Dense(128, activation='relu', name='merged_feedforward_1')(concat_layer)
    x = Dense(64, activation='relu', name='merged_feedforward_2')(x)
    
    possibility_outputs = Dense(1, activation='sigmoid', name='label_output')(x)  # softmax  sigmoid
    
    adam = Adam(lr=learning_rate, decay= adam_decay)  # , clipnorm=0.5
    model = Model(inputs=[content_input, stats_input], outputs=[possibility_outputs])  # stats_input
    model.compile(loss=binary_crossentropy_add_mi, optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    return model

# skopt
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, hidden_head, multiheads,
            epochs, batch_size, adam_decay):

    model = Multi_Head_Attention_sematic_stats(
                         X_train=X_train,
                         y_train=y_train,
                         learning_rate=learning_rate,
                         hidden_head=hidden_head,
                         multiheads=multiheads,
                         adam_decay=adam_decay
                        )
    #named blackbox becuase it represents the structure
    blackbox = model.fit(x=X_train,
                        y=y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.15,
                        shuffle=True,
                        )  # X_train, [y_train]
    #return the validation accuracy for the last epoch.
    accuracy = blackbox.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.3%}".format(accuracy))
    print()


    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    tensorflow.reset_default_graph()
    
    return -accuracy


K.clear_session()
tensorflow.reset_default_graph()
gp_result = gp_minimize(func=fitness,
                    dimensions=dimensions,
                    n_calls=12,
                    noise= 0.01,
                    n_jobs=-1,
                    kappa = 5,
                    x0=default_parameters)

print("best accuracy was " + str(round(gp_result.fun *-100,2))+"%.")
print(gp_result.x)
print(gp_result.func_vals)


"""
# no_footnotes-0, primary_sources-1, refimprove-2, original_research-3, advert-4, notability-5
# dim_learning_rate, dim_hidden_rnn, dropout_rate, dim_epochs, dim_batch_size, dim_adam_decay
content-bilstm
params = [
        [],
        []，
        [],
        [],
        [],
        []
        ]
"""
