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
from keras import regularizers, optimizers
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.engine.topology import Layer
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Bidirectional, Flatten, Dropout, Multiply, Add, Permute, concatenate
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
# from keras_layer_normalization import LayerNormalization

from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

from evaluation_metrics import  precision, recall, fbeta_score, fmeasure, getAccuracy


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


def Multi_Head_Attention_sematic_stats(X_train_content, X_train_stats, y_train, X_val_content, X_val_stats, y_val, learning_rate, adam_decay, hidden_head, multiheads, batch_size, epochs):
    # content part
    multiheads = multiheads
    head_dim = hidden_head
    content_input = Input(shape=(X_train_content.shape[1],X_train_content.shape[2]), name='content_bert_input')
    embedding = Position_Embedding()(content_input)
    attention = Attention(multiheads=multiheads,head_dim=head_dim,mask_right=False)([embedding,embedding,embedding])
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
    
    adam = optimizers.Adam(lr=learning_rate, decay=adam_decay)  # , clipnorm=0.5
    model = Model(inputs=[content_input, stats_input], outputs=[possibility_outputs])  # stats_input
    model.compile(loss=binary_crossentropy_add_mi, optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    history = model.fit([X_train_content, X_train_stats], [y_train], batch_size, epochs, validation_data=([X_val_content, X_val_stats], [y_val]), shuffle=True, callbacks=[TensorBoard(log_dir='./tmp/log')])

    return model, history


def draw_graph(graph_type):
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history[graph_type])
    plt.plot(history.history['val_' + graph_type])
    plt.title('Model ' + graph_type)
    plt.ylabel(graph_type)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    # 输入限制，文章数量及每篇文章输入的句子/段落数
    file_num_limit = 45614  # total 45614
    paras_limit=20

    # params get through skopt
    params = [[0.0008220285540122005, 47, 62, 21, 205, 0.009459087477038792],
        [0.0003612832710858312, 30, 17, 21, 35, 0.0046708384009473535],
        [0.0003006378046657311, 48, 29, 21, 81, 0.007097605957309866],
        [0.0004701490614629333, 39, 38, 23, 210, 2.7957469250045207e-05],
        [0.00042665320871070056, 29, 24, 19, 72, 0.006843620057362327],
        [0.0001, 32, 32, 20, 100, 1e-06]]

    encoded_contents, onehotlabels, stats_features = prepare_input(file_num_limit, paras_limit)

    # stats_features 标准化
    scaler = preprocessing.StandardScaler() #实例化
    scaler = scaler.fit(stats_features)
    stats_features = scaler.transform(stats_features)
    
    # 换算成二分类
    # no_footnotes-0, primary_sources-1, refimprove-2, original_research-3, advert-4, notability-5
    flaw_evaluation = []
    for flaw_index in range(3,4):
        no_good_flaw_type = flaw_index # finished
        # 找出FA类的索引
        FA_indexs = [index for index in range(len(onehotlabels)) if sum([int(item) for item in onehotlabels[index]]) == 0]
        # 找出二分类另外一类的索引
        not_good_indexs = [index for index in range(len(onehotlabels)) if onehotlabels[index][no_good_flaw_type] > 0]
        binary_classification_indexs = FA_indexs + not_good_indexs
        print('FA count:', len(FA_indexs), 'no good count:', len(not_good_indexs))
        X_contents = np.array([encoded_contents[index,:,:] for index in binary_classification_indexs])
        y_train = np.array([onehotlabels[index] for index in binary_classification_indexs])
        X_stats = np.array([stats_features[index] for index in binary_classification_indexs])
        
        # 变成二分类的标签
        y_train = y_train[:,no_good_flaw_type]
        y_train = np.array([[label] for label in y_train])
        ### split data into training set and label set
        # X_train, X_test, y_train, y_test = train_test_split(encoded_contents, onehotlabels, test_size=0.1, random_state=42)
        ### params set choose
        target_param = params[no_good_flaw_type]

        learning_rate = target_param[0]
        hidden_head = target_param[1]
        multiheads = target_param[2]
        epochs = target_param[3]
        batch_size = target_param[4]
        adam_decay = target_param[5]
        ### create the deep learning models
        # 训练模型
        X_train_content, X_train_stats, y_train = X_contents, X_stats, y_train

        # 引入十折交叉验证
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        kfold_precision, kfold_recall, kfold_f1_score, kfold_acc, kfold_TNR = [], [], [], [], []
        fold_counter = 0
        for train, test in kfold.split(X_train_content, y_train):
            print('folder comes to:', fold_counter)
            _precision, _recall, _f1_score, _acc, _TNR = 0, 0, 0, 0, 0
            X_test_content_kfold, X_test_stats_kfold, y_test_kfold = X_train_content[test], X_train_stats[test], y_train[test]
            X_val_content_kfold, X_val_stats_kfold, y_val_kfold = X_train_content[train[-1000:]], X_train_stats[train[-1000:]], y_train[train[-1000:]]
            X_train_content_kfold, X_train_stats_kfold, y_train_kfold = X_train_content[train[:-1000]], X_train_stats[train[:-1000]], y_train[train[:-1000]]

            # 采用后1000条做验证集
            # X_val, y_val = X_train[-1000:], y_train[-1000:]
            # X_train, y_train = X_train[:-1000], y_train[:-1000]
            model, history = Multi_Head_Attention_sematic_stats(X_train_content_kfold, X_train_stats_kfold, y_train_kfold, 
                                                                X_val_content_kfold, X_val_stats_kfold, y_val_kfold, 
                                                                learning_rate, adam_decay, hidden_head, multiheads, batch_size, epochs)
            prediction = model.predict([X_test_content_kfold, X_test_stats_kfold])  # {'content_bert_input': X_test_content, 'stats_input': X_test_stats}
            _precision, _recall, _f1_score, _acc, _TNR = getAccuracy(prediction, y_test_kfold)
            print('precision:', _precision, 'recall', _recall, 'f1_score', _f1_score, 'accuracy', _acc, 'TNR', _TNR)
            kfold_precision.append(_precision)
            kfold_recall.append(_recall)
            kfold_f1_score.append(_f1_score)
            kfold_acc.append(_acc)
            kfold_TNR.append(_TNR)
            fold_counter += 1
            # Delete the Keras model with these hyper-parameters from memory.
            del model
    
            # Clear the Keras session, otherwise it will keep adding new
            # models to the same TensorFlow graph each time we create
            # a model with a different set of hyper-parameters.
            K.clear_session()
            tensorflow.reset_default_graph()
        print('10 k average evaluation is:', 'precision:', np.mean(kfold_precision), 'recall', np.mean(kfold_recall), 'f1_score', np.mean(kfold_f1_score), 'accuracy', np.mean(kfold_acc), 'TNR', np.mean(kfold_TNR))

        evaluation_value = str(no_good_flaw_type) + ' 10 k average evaluation is: ' + ' precision: ' + str(np.mean(kfold_precision)) + ' recall ' + str(np.mean(kfold_recall)) + ' f1_score ' + str(np.mean(kfold_f1_score)) + ' accuracy ' + str(np.mean(kfold_acc)) + ' TNR ' + str(np.mean(kfold_TNR))
        flaw_evaluation.append(evaluation_value)

    for item in flaw_evaluation:
        print(item)

