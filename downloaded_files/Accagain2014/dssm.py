#coding=utf-8

import sys
sys.path.append('../helper/')
import time

import numpy as np
import tensorflow as tf

import tools


class DSSM(object):
    '''
        Impletement DSSM Model in the Paper:  Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
    '''
    def __init__(self, hash_tokens_nums=3000, dnn_layer_nums=1, dnn_hidden_node_nums=50, feature_nums=50,
                batch_size=10, neg_nums=4, learning_rate=0.5, max_epochs=200, loss_kind='mcl', w_init=0.1, \
                 save_model_path='./', mlp_hidden_node_nums=32, mlp_layer_nums=2, input_is_sparse=False):
        '''
            paras:
                hash_tokens_nums: word hash后词的个数
                dnn_layer_nums: dnn的层数
                dnn_hidden_node_nums: dnn的结点个数
                feature_nums: 最终输出的特征的个数
                batch_size: 每个batch的大小
                neg_nums: 负样本的个数
                learning_rate: 学习率
                max_epoch: 迭代次数
                loss_kind: 'mcl': maximize the condition likelihood，极大似然估计条件概率; 'log_loss'：交叉熵的方式计算loss
                w_init: 权重初始化
                save_model_path: 保存验证集上最优模型的文件路劲
                mlp_hidden_node_nums: 学习到的隐向量连接后加mlp层的节点数
                mlp_layer_nums： mlp层的层数
                input_is_sparse: 输入是否是sparse矩阵
        '''

        self.hash_token_nums = hash_tokens_nums
        self.dnn_layer_nums = dnn_layer_nums
        self.dnn_hidden_node_nums = dnn_hidden_node_nums
        self.feature_nums = feature_nums
        self.batch_size = batch_size
        self.neg_nums = neg_nums
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.loss_kind = loss_kind
        self.positive_weights = 1
        self.w_init = w_init
        self.save_model_path = save_model_path
        self.mlp_hidden_node_nums = mlp_hidden_node_nums
        self.mlp_layer_nums = mlp_layer_nums
        self.input_is_sparse = input_is_sparse

        '''
            query and doc 使用不同的网络结构，像论文中提到的那样
        '''
        if not self.input_is_sparse:
            self.input_q = tf.placeholder(tf.float32, shape=[None, self.hash_token_nums]) # sample_nums, word_nums, hash_tokens_nums
            self.input_doc = tf.placeholder(tf.float32, shape=[None, self.hash_token_nums]) # sample_nums, word_nums, hash_tokens_nums
        else:
            self.input_q = tf.sparse_placeholder(tf.float32, shape=[None, self.hash_token_nums])
            self.input_doc = tf.sparse_placeholder(tf.float32, shape=[None, self.hash_token_nums])
            
        self.label = tf.placeholder(tf.float32, shape=[None])

        self.predict_doc = None
        self.predict_query = None

        self.relevance = self.create_model_op()

        if self.loss_kind == 'mlc':
            self.loss = self.create_loss_max_condition_lh_op()
        elif self.loss_kind == 'log_loss':
            self.loss = self.create_log_loss_op()
        else:
            pass

        self.train = self.create_train_op()

    def set_positive_weights(self, positive_weights):
        self.positive_weights = positive_weights

    def create_model_op(self):

        '''
            建立整个模型，分成两端的网络，query端和doc端的
        '''

        features = []
        structures = ['query_dnn', 'doc_dnn']
        input_dict = {
            structures[0]: self.input_q,
            structures[1]: self.input_doc
        }
        
        '''
            尝试用一种结构试下
        '''
        
        result = [0] * 2
        with tf.variable_scope('DNN'):
            now_w_init = tools.xavier_init(self.hash_token_nums, self.dnn_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.hash_token_nums, self.dnn_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer1')
            b = tf.Variable(tf.zeros([self.dnn_hidden_node_nums]), name="bias_DNN_layer1")
            
            result[0] = input_dict['query_dnn']
            result[1] = input_dict['doc_dnn']
            
            if self.input_is_sparse:
                result[0] = tf.sparse_tensor_dense_matmul(result[0], w) + b
                result[1] = tf.sparse_tensor_dense_matmul(result[1], w) + b
                
            else:
                result[0] = tf.matmul(result[0], w) + b
                result[1] = tf.matmul(result[1], w) + b
            
            result[0] = tf.nn.tanh(result[0])
            result[1] = tf.nn.tanh(result[1])
            
            
            now_w_init = tools.xavier_init(self.dnn_hidden_node_nums, self.dnn_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.dnn_hidden_node_nums, self.dnn_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer2')
            b = tf.Variable(tf.zeros([self.dnn_hidden_node_nums]), name="bias_DNN_layer2")
            result[0] = tf.matmul(result[0], w) + b
            result[0] = tf.nn.tanh(result[0])
            result[1] = tf.matmul(result[1], w) + b
            result[1] = tf.nn.tanh(result[1])
            
            
            now_w_init = tools.xavier_init(self.dnn_hidden_node_nums, self.dnn_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.dnn_hidden_node_nums, self.dnn_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer3')
            b = tf.Variable(tf.zeros([self.dnn_hidden_node_nums]), name="bias_DNN_layer3")
            result[0] = tf.matmul(result[0], w) + b
            result[0] = tf.nn.tanh(result[0])
            result[1] = tf.matmul(result[1], w) + b
            result[1] = tf.nn.tanh(result[1])
            
            now_w_init = tools.xavier_init(self.dnn_hidden_node_nums, self.dnn_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.dnn_hidden_node_nums, self.dnn_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer4')
            b = tf.Variable(tf.zeros([self.dnn_hidden_node_nums]), name="bias_DNN_layer4")
            result[0] = tf.matmul(result[0], w) + b
            result[0] = tf.nn.tanh(result[0])
            result[1] = tf.matmul(result[1], w) + b
            result[1] = tf.nn.tanh(result[1])
            '''
            
            now_w_init = tools.xavier_init(self.dnn_hidden_node_nums, self.dnn_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.dnn_hidden_node_nums, self.dnn_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer5')
            b = tf.Variable(tf.zeros([self.dnn_hidden_node_nums]), name="bias_DNN_layer5")
            result[0] = tf.matmul(result[0], w) + b
            result[0] = tf.nn.tanh(result[0])
            result[1] = tf.matmul(result[1], w) + b
            result[1] = tf.nn.tanh(result[1])
            '''
            
            
            now_w_init = tools.xavier_init(self.dnn_hidden_node_nums, self.feature_nums)
            w = tf.Variable(
                        tf.random_uniform([self.dnn_hidden_node_nums, self.feature_nums], -now_w_init, now_w_init), name='weights_DNN_layer_last')
            b = tf.Variable(tf.zeros([self.feature_nums]), name="bias_DNN_layer_last")
            result[0] = tf.matmul(result[0], w) + b
            result[0] = tf.nn.tanh(result[0])
            result[1] = tf.matmul(result[1], w) + b
            result[1] = tf.nn.tanh(result[1])
            
            
            '''
            i = tf.constant(0)
            sum_layer = self.dnn_layer_nums
            #node_nums = tf.convert_to_tensor([self.dnn_hidden_node_nums] * self.dnn_layer_nums + [self.dnn_hidden_node_nums])
            node_nums = [self.dnn_hidden_node_nums] * self.dnn_layer_nums + [self.dnn_hidden_node_nums]
            
            cond = lambda x, layer, result: tf.less(x, sum_layer)
            layer = 0
            def body(i, layer, result):
                tmp = tf.add(i, 1)
                w = tf.Variable(
                        tf.random_uniform([node_nums[layer], node_nums[layer+1]], -self.w_init, self.w_init))
                b = tf.Variable(tf.zeros([node_nums[layer+1]]))
               
                result[0] = tf.matmul(result[0], w) + b
                result[0] = tf.nn.tanh(result[0])
                result[1] = tf.matmul(result[1], w) + b
                result[1] = tf.nn.tanh(result[1])
                    
                return tmp, layer, result
                
            i, _, result = tf.while_loop(cond, body, [i, layer, result])
            '''   
            
            features.append(result[0])
            features.append(result[1])
        

        self.predict_query = features[0]
        self.predict_doc = features[1]

        '''
            为了对学习到了两个向量进行相似度打分，加一个mlp层, 最后一层全连接

        '''

        result = tf.concat(features, -1)

        with tf.variable_scope('mlp'):
            node_nums = tf.convert_to_tensor([self.feature_nums*2] + [self.mlp_hidden_node_nums] * self.mlp_layer_nums + [1])
            sum_layer = self.mlp_hidden_node_nums + 1
            
            
            now_w_init = tools.xavier_init(self.feature_nums * 2, self.mlp_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.feature_nums*2, self.mlp_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer1')
            b = tf.Variable(tf.zeros([self.mlp_hidden_node_nums]), name="bias_DNN_layer1")
            result = tf.matmul(result, w) + b
            result = tf.nn.tanh(result)
            
            '''
            now_w_init = tools.xavier_init(self.mlp_hidden_node_nums, self.mlp_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.mlp_hidden_node_nums, self.mlp_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer2')
            b = tf.Variable(tf.zeros([self.mlp_hidden_node_nums]), name="bias_DNN_layer2")
            result = tf.matmul(result, w) + b
            result = tf.nn.tanh(result)
            
            now_w_init = tools.xavier_init(self.mlp_hidden_node_nums, self.mlp_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.mlp_hidden_node_nums, self.mlp_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer2')
            b = tf.Variable(tf.zeros([self.mlp_hidden_node_nums]), name="bias_DNN_layer2")
            result = tf.matmul(result, w) + b
            result = tf.nn.tanh(result)
            
            now_w_init = tools.xavier_init(self.mlp_hidden_node_nums, self.mlp_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.mlp_hidden_node_nums, self.mlp_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer3')
            b = tf.Variable(tf.zeros([self.mlp_hidden_node_nums]), name="bias_DNN_layer3")
            result = tf.matmul(result, w) + b
            result = tf.nn.tanh(result)
            
            now_w_init = tools.xavier_init(self.mlp_hidden_node_nums, self.mlp_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.mlp_hidden_node_nums, self.mlp_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer4')
            b = tf.Variable(tf.zeros([self.mlp_hidden_node_nums]), name="bias_DNN_layer4")
            result = tf.matmul(result, w) + b
            result = tf.nn.tanh(result)
            
            now_w_init = tools.xavier_init(self.mlp_hidden_node_nums, self.mlp_hidden_node_nums)
            w = tf.Variable(
                        tf.random_uniform([self.mlp_hidden_node_nums, self.mlp_hidden_node_nums], -now_w_init, now_w_init), name='weights_DNN_layer5')
            b = tf.Variable(tf.zeros([self.mlp_hidden_node_nums]), name="bias_DNN_layer5")
            result = tf.matmul(result, w) + b
            result = tf.nn.tanh(result)
            '''
            
            now_w_init = tools.xavier_init(self.mlp_hidden_node_nums, 1)
            w = tf.Variable(
                        tf.random_uniform([self.mlp_hidden_node_nums, 1], -now_w_init, now_w_init), name='weights_DNN_layer_last')
            b = tf.Variable(tf.zeros([1]), name="bias_DNN_layer_last")
            result = tf.matmul(result, w) + b
            result = tf.nn.sigmoid(result)
            

        # norms1 = tf.sqrt(tf.reduce_sum(tf.square(features[0]), 1, keep_dims=False))
        # norms2 = tf.sqrt(tf.reduce_sum(tf.square(features[1]), 1, keep_dims=False))
        # relevance =  tf.reduce_sum(features[0] * features[1], 1) / norms1 / norms2

        # w_r = tf.Variable(tf.random_uniform([1], -self.w_init, self.w_init), name="weight-of-relevance")
        # b_r = tf.Variable(tf.zeros([1]), name="bais-of-relevance")
        # relevance = relevance * w_r + b_r
        # relevance = tf.nn.softmax(relevance)

        return tf.reshape(result, [-1])


    def create_loss_max_condition_lh_op(self):
        '''
            用极大似然的方法计算, 正例的条件概率
            计算相关文档的loss, gama经验值也用来学习
        :return:
        '''
        gama = tf.Variable(tf.random_uniform([1]), name="gama")
        ret = self.relevance * gama
        ret = tf.reshape(ret, [-1, self.neg_nums+1])
        ret = tf.log(tf.nn.softmax(ret))
        ret = tf.reduce_sum(ret, 0) # 行相加
        return -tf.gather(ret, 0) # 得到第一个，也即是正例的loss


    def create_log_loss_op(self):
        '''
            计算log_loss, 也就是交叉熵
        :return:
        '''
        return tf.reduce_sum(tf.contrib.losses.log_loss(self.relevance, self.label))


    def create_train_op(self):
        '''
            采用梯度下降方式学习
        :return:
        '''
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        #return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)


    def creat_feed_dict(self, query_batch, doc_batch, label_batch):
        '''
        :param query_batch: 查询输入
        :param doc_batch: 文档输入
        :param label_batch: 查询和文档对应的相关性label
        :return:
        '''

        if self.input_is_sparse:
            query_coo_matrix = query_batch.tocoo()
            doc_coo_matrix = doc_batch.tocoo()
            self.feed_dict = {
                self.input_q : tf.SparseTensorValue(np.array([query_coo_matrix.row, query_coo_matrix.col]).T, query_batch.data, query_batch.shape),
                self.input_doc : tf.SparseTensorValue(np.array([doc_coo_matrix.row, doc_coo_matrix.col]).T, doc_batch.data, doc_batch.shape),
                self.label : label_batch
            }
        else:
            self.feed_dict = {
                self.input_q : query_batch,
                self.input_doc: doc_batch,
                self.label : label_batch
            }


    def run_epoch(self, sess, query_input, doc_input, labels, is_valid=False):
        '''
        计算一次迭代过程
        :param sess:
        :param query_input:
        :param doc_input:
        :param labels:
        :return:
        '''
        average_loss = 0
        step = 0
        relevance = []
      
        for step, (query, doc, label) in enumerate(
                tools.data_iterator(query_input, doc_input, labels, self.batch_size, shuffle=True, is_normalize=True)
            ):
            # print query[1, 1], doc[1, 1], label[1]
            self.creat_feed_dict(query, doc, label)
            #print query.shape, doc.shape, label.shape
            #print type(query),is_sparse
            # self.set_positive_weights(len(query))

            # shape1, shape2, shape3 = sess.run([self.shape_1, self.shape_2, self.shape_3], feed_dict=self.feed_dict)
            # print shape1, shape2, shape3

            if not is_valid:
                # 跑这个train的时候 才更新W
                _, loss_value, predict_query, predict_doc, relevance = sess.run([self.train, self.loss, self.predict_query\
                        , self.predict_doc, self.relevance], feed_dict=self.feed_dict)
            else:

                loss_value, relevance = sess.run([self.loss, self.relevance], feed_dict=self.feed_dict)
                # print 'Chcek ', sklearn.metrics.log_loss(label, relevance), loss_value

            average_loss += loss_value
            #print 'step ', step, loss_value
            #print 'predict ', predict_query[0], predict_doc[0], relevance[0]
        return average_loss / (step+1), relevance


    def fit(self, sess, query_input, doc_input, labels, valid_q_input=None, valid_d_input=None, valid_labels=None, \
            load_model=False):
        '''
        模型入口
        :param sess:
        :param query_input:
        :param doc_input:
        :param labels:
        :return:
        '''
        losses = []
        best_loss = 99999
        saver = tf.train.Saver()
        if load_model:
            saver.restore(sess, self.save_model_path)
            start_time = time.time()
            valid_loss, _ = self.run_epoch(sess, valid_q_input, valid_d_input, valid_labels, is_valid=True)
            duration = time.time() - start_time
            print('valid loss = %.5f (%.3f sec)'
                  % (valid_loss, duration))
            losses.append(valid_loss)
            return losses

        for epoch in range(self.max_epochs):
            start_time = time.time()
            average_loss, relevance = self.run_epoch(sess, query_input, doc_input, labels)
            duration = time.time() - start_time

            if (epoch+1) % 1 == 0:
                if valid_labels is None:
                    print('Epoch %d: loss = %.5f relevance[0] = %.5f (%.3f sec)'
                        % (epoch+1, average_loss, relevance[0], duration))
                else:
                    valid_loss, _ = self.run_epoch(sess, valid_q_input, valid_d_input, valid_labels, is_valid=True)
                    if valid_loss < best_loss:
                        print 'Save model'
                        best_loss = valid_loss
                        saver.save(sess, self.save_model_path)
                    duration = time.time() - start_time
                    print('Epoch %d: loss = %.5f valid loss = %.5f relevance[0] = %.5f (%.3f sec)'
                          % (epoch+1, average_loss, valid_loss, relevance[0], duration))
                    sys.stdout.flush()
            losses.append(average_loss)

        if not valid_labels is None:
            print 'Final valid loss: ', best_loss
        return losses

    def predict(self, sess, query, doc, labels):
        '''
        计算预测过后的查询与文档的相关性
        :param sess:
        :param query:
        :param doc:
        :param labels:
        :return:
        '''
        if not self.is_sparse:
            self.creat_feed_dict(query, doc, labels)
            predict = sess.run(self.relevance, feed_dict=self.feed_dict)
        else:
            predict = []
            for step, (query_, doc_, label_) in enumerate(
                tools.data_iterator(query, doc, labels, self.batch_size, shuffle=True, is_normalize=True)
            ):
                self.creat_feed_dict(query, doc, labels)
                now_pre = sess.run(self.relevance, feed_dict=self.feed_dict)
                predict += now_pre
                
        return predict



def test_dssm():
    '''
    测试函数
    :return:
    '''
    with tf.Graph().as_default():
        tf.set_random_seed(1)

        model = DSSM(hash_tokens_nums=30000, dnn_layer_nums=2, dnn_hidden_node_nums=300, feature_nums=128,
                batch_size=10, neg_nums=4, learning_rate=0.02, max_epochs=500)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        np.random.seed(1)

        query = np.random.rand(500, 30000)
        doc = np.random.rand(500, 30000)
        label = np.array([1, 0, 0, 0, 0] * 100)

        #print query
        #print doc
        #print label

        losses = model.fit(sess, query, doc, label)

    #print losses[-1]


if __name__ == '__main__':
    test_dssm()
