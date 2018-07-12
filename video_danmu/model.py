import tensorflow as tf
import config
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name='W')


def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name="bias")


class Model:
    def __init__(self, data, y, keep_prob):
        self.data = data
        self.modality = config.modality
        self.target = y
        self.lr = config.lr
        self.n_input_frame = config.n_input_frame
        self.n_input_danmu = config.doc_vector_size
        self.n_output_video = config.n_output_modality[0]
        self.n_output_danmu = config.n_output_modality[1]
        self.n_input_classification = 0
        self.n_part_danmu = config.n_part_danmu
        self.n_part_video = config.frame_num
        if self.modality[0] == 1:
            self.n_input_classification += self.n_output_video
        if self.modality[1] == 1:
            self.n_input_classification += self.n_output_danmu
        self.n_hidden_lstm_video = config.n_hidden_lstm_video
        self.n_hidden_lstm_danmu = config.n_hidden_lstm_danmu
        self.n_classes = config.n_classes
        self.status = config.train_status
        self.frame_num = config.frame_num
        self.video_output = None
        self.danmaku_output = None
        self.video_output_reconstruct = None
        self.danmaku_output_reconstruct = None
        self.keep_prob = keep_prob
        self.encode = None
        # build the net
        if config.train_status == 0:
            # unsupervised part : AutoEncoder
            self.build_unsupervised_net()
        else:
            # supervised part : LSTM
            self.build_supervised_net()
        # sum the loss
        self.loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
        self.add_optimize()
        self.add_checkpoint()
        # self._graph = None

    def add_layer(self, inputs, in_size, out_size, layer_name, activity_func=None):
        '''
        config:
            inputs: 层的输入
            in_size: 输入的shape
            out_size: 输出的shape, 与in_size共同决定了权重的shape
            activity_fuc: 激活函数
        '''
        # 正太分布下，初始化权重W
        # W = tf.Variable(tf.random_uniform([in_size, out_size], -1.0, 1.0), name="W", dtype = tf.float32)
        W = tf.get_variable(
            "W" + layer_name, shape=[in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        # 偏置一般用一个常数来初始化就行
        bias = tf.Variable(tf.constant(
            0.1, shape=[out_size]), name="bias", dtype=tf.float32)
        # Wx_Plus_b = tf.matmul(inputs, W) + bias 这种方式与下面的均可以
        Wx_Plus_b = tf.nn.xw_plus_b(inputs, W, bias)
        if activity_func is None:
            outputs = Wx_Plus_b
        else:
            outputs = activity_func(Wx_Plus_b)
        return outputs  # 返回的是该层的输出

    def compute_lstm_vector(self, lstm_input, time_steps, feature_size, n_hidden_lstm):
        """compute the lstm network last hidden state

        Args:
            lstm_input: the CNN feature, also the input of the LSTM part, [batch_size, time_steps, feature_size]
        Returns:
            final_state: the final hidden state of the LSTM part
        """
        # in tensorflow rc1.1, the following should work, but in rc1.2, it doesn't work. Maybe the tensorflow bugs.
        # frame_list = [tf.squeeze(input) for input in tf.split(value=fc_7, num_or_size_splits=self.frame_num, axis=1)]

        print('LSTM Input', lstm_input, feature_size)
        time_first_input = tf.transpose(lstm_input, [1, 0, 2])
        print('LSTM Input', time_first_input, feature_size)
        reshaped_input = tf.reshape(time_first_input, [-1, feature_size])
        input_list = tf.split(reshaped_input, time_steps, axis=0)
        # here utilize the layernorm LSTM cell to speed the convergence of the training
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            n_hidden_lstm, forget_bias=0.1)
        outputs, output_state = tf.contrib.rnn.static_rnn(
            cell, inputs=input_list, dtype=tf.float32)
        # take the last state of all the states
        final_state = outputs[-1]

        return final_state

    def build_supervised_net(self):
        """build the supervise net.

        include two parts: the video part and the danmu part
        """

        # if not(self._graph is None):
        #     return self._graph
        feature_fusion = []
        # wangjialin 2017.6.7 refractor codes
        # use the same lstm function with video lstm part
        # video LSTM
        re_1 = tf.contrib.layers.l1_regularizer(0.001)
        # re_1 = None
        re_2 = tf.contrib.layers.l2_regularizer(0.001)
        # re_2 = None
        if config.unsupervised_network == 'DistAE':
            L = len(config.DistAE_video_hidden_size_list)
            feature_size_video = config.DistAE_video_hidden_size_list[L // 2]
            feature_size_danmu = config.DistAE_danmu_hidden_size_list[L // 2]
        elif config.unsupervised_network == 'DCCAE':
            L = len(config.DCCAE_video_hidden_size_list)
            feature_size_video = config.DCCAE_video_hidden_size_list[L // 2]
            feature_size_danmu = config.DCCAE_danmu_hidden_size_list[L // 2]
        elif config.unsupervised_network == 'no':
            feature_size_video = self.n_input_frame
            feature_size_danmu = self.n_input_danmu

        print('feature_size_danmu', feature_size_danmu)
        print('config.n_input_danmu', config.n_input_danmu)
        print('config.doc_vector_size', config.doc_vector_size, flush=True)
        if self.modality[0] == 1:
            print('The video modality is used')
            with tf.variable_scope("video"):
                video_output = self.compute_lstm_vector(
                    self.data[0], time_steps=self.n_part_danmu, feature_size=feature_size_video,  n_hidden_lstm=config.n_output_modality[0])
            if feature_fusion == []:
                feature_fusion = video_output
            else:
                feature_fusion = tf.concat([feature_fusion, video_output], 1)
        # danmu LSTM
        if self.modality[1] == 1:
            print('The danmu modality is used')
            with tf.variable_scope('danmuku'):
                danmuku_output = self.compute_lstm_vector(
                    self.data[1], time_steps=self.n_part_danmu, feature_size=feature_size_danmu, n_hidden_lstm=config.n_output_modality[1])
                # feature fusion
            if feature_fusion == []:
                feature_fusion = danmuku_output
            else:
                feature_fusion = tf.concat([feature_fusion, danmuku_output], 1)

        with tf.variable_scope('classification'):
            # bn_1 = tf.layers.batch_normalization(feature_fusion)
            # dp_1 = tf.layers.dropout(bn_1, rate=self.keep_prob)
            l1_class = self.add_layer(feature_fusion, self.n_input_classification,
                                      config.n_hidden_classification_1, 'hidden_layer_1', activity_func=tf.nn.relu)
            bn_2 = tf.layers.batch_normalization(l1_class)
            dp_2 = tf.layers.dropout(bn_2, rate=self.keep_prob)
            # l2_class = self.add_layer(l1_class, config.n_hidden_classification_1, config.n_hidden_classification_2, 'hidden_layer_2', activity_func=tf.nn.tanh)
            # bn_3 = tf.layers.batch_normalization(l2_class)
            # dp_3 = tf.layers.dropout(bn_3, rate = self.keep_prob)
            l3_class = self.add_layer(dp_2, config.n_hidden_classification_2,
                                      self.n_classes, 'prediction_layer', activity_func=tf.nn.relu)

        self._graph = l3_class
        self.prob = tf.nn.softmax(l3_class)
        self.prediction = tf.argmax(self.prob, 1)
        regularization_loss = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.add_to_collection("losses", tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.prob, labels=self.target)) + regularization_loss)

    # @property
    # def accuracy(self):
    #     print('Tensorflow accuracy')
    #     if self._accuracy is None:
    #         self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.target, 1), self.prediction), tf.float32))
    #     return self._accuracy

    def initial_AE_weights(self, name, hidden_size):
        """initial the weights in the AutoEncoders

        suppose AE has L hidden layers and L reconstruction layers,
        then length of hidden_size should be (2*L+2), add the input size and the output size

        Args:
            name : the name of the network, like "video_AE", "danmu_AE"
            hidden_size: list, include the size of every data
        Returns:
            None
        """

        self.__dict__[name] = {}
        for i in range(len(hidden_size) - 1):
            w_name = name + "_hidden_" + str(i + 1) + "_w"
            self.__dict__[name][w_name] = tf.get_variable(name=w_name, shape=[
                                                          hidden_size[i], hidden_size[i + 1]], initializer=tf.contrib.layers.xavier_initializer())

            print('Initialize', self.__dict__[name][w_name])
            b_name = name + "_hidden_" + str(i + 1) + "_b"
            self.__dict__[name][b_name] = tf.get_variable(
                name=b_name, shape=[hidden_size[i + 1]], initializer=tf.constant_initializer(0.1))

    def build_AE(self, inputs,  name,  L):
        """ build the AutoEncoder

        the AutoEncoder : the Encoder and the Decoder has the same number of layers, and are in MLP style

        Args:
            input : input tensor, shape:[None, n_input_size]
            name : string, used for get the weights of the net
            L : the number of the layers, include the input layer and the output layer
        Returns:
            encode : the encode of the input, is the middle element of the layer_out
            decode : the output of the AutoEncoder, the last element of the layer_out
        """

        layer_out = [inputs]  # store the data of all the layers
        for i in range(L - 1):
            w_name = name + "_hidden_" + str(i + 1) + "_w"
            b_name = name + "_hidden_" + str(i + 1) + "_b"
            w = self.__dict__[name][w_name]
            print('W', i, w)
            b = self.__dict__[name][b_name]
            # MLP, Wx_Plus_b
            layer_out.append(tf.nn.relu(tf.matmul(layer_out[-1], w) + b))
            # if i == 0 and self.encode == None:
            #     self.w = w
            #     self.encode = tf.matmul(inputs, w) + b
        encode = layer_out[L // 2]
        decode = layer_out[L - 1]
        return encode, decode

    def build_unsupervised_net(self):
        """ build the unsupervised net, such as DistAE, DCCAE

        refer to the paper: On Deep Multi-View Representation Learning.
        for better understanding, plz read the paper :)
        """

        # loss = self.DistAE()
        loss = self.DCCAE()
        tf.add_to_collection("losses", loss)

    def DistAE(self):
        """the DistAE implementation, used for unsupervied part

        Note : the hyper-parameter: the trade-off AE_lambda and the hidden size
        """

        # video_input : [batch_size, n_input_frame]
        # danmu_input : [batch_size, n_input_danmu]
        video_input = tf.reshape(self.data[0], [-1, self.n_input_frame])
        danmu_input = tf.reshape(self.data[1], [-1, self.n_input_danmu])
        video_hidden_size_list = config.DistAE_video_hidden_size_list
        danmu_hidden_size_list = config.DistAE_danmu_hidden_size_list

        self.video_input = video_input
        self.danmu_input = danmu_input
        re_2 = tf.contrib.layers.l2_regularizer(0.01)
        re_2 = None
        # The f(x), g(y) should have the same dimension
        # TODO: Add Noise
        with tf.variable_scope("video_AE", regularizer=re_2):
            # initial weights
            self.initial_AE_weights(
                name="video_AE", hidden_size=video_hidden_size_list)
            video_AE_f, video_AE_p_f = self.build_AE(
                video_input, name="video_AE", L=len(video_hidden_size_list))
            # tf.get_variable_scope().reuse_variables()

        # TODO: Add Noise
        with tf.variable_scope("danmu_AE"):
            # initial weights
            self.initial_AE_weights(
                name="danmu_AE", hidden_size=danmu_hidden_size_list)
            # danmu_AE_g, danmu_AE_q_g = self.build_AE(danmu_input, name="danmu_AE", L=len(danmu_hidden_size_list))
            danmu_AE_g, danmu_AE_q_g = danmu_input, danmu_input
            # tf.get_variable_scope().reuse_va  riables()

        self.encoder_video = video_AE_f
        self.decoder_video = video_AE_p_f

        # self.encoder_danmu = danmu_input
        # self.decoder_danmu = danmu_input

        self.encoder_danmu = danmu_AE_g
        self.decoder_danmu = danmu_AE_q_g

        # self.encoder_danmu = self.danmu_input
        # self.decoder_danmu = self.danmu_input

        # loss function
        # f(x): video_AE_hidden_3 , g(y) : danmu_AE_hidden_3
        # p(f(x)): video_final_reconstruct, q(g(y)): danmu_final_reconstruct
        loss_part_1_numerator = tf.reduce_sum(
            tf.square(tf.subtract(video_AE_f, danmu_AE_g)))
        loss_part_1_denominator = tf.reduce_sum(
            tf.square(video_AE_f)) + tf.reduce_sum(tf.square(danmu_AE_g))
        loss_part_1 = loss_part_1_numerator / loss_part_1_denominator
        loss_part_2 = (tf.reduce_sum(tf.square(tf.subtract(
            video_input, video_AE_p_f)))) * config.DistAE_lambda
        #  + tf.reduce_sum(tf.square(tf.subtract(danmu_input, danmu_AE_q_g)))) * config.DistAE_lambda
        # )*config.DistAE_lambda
        regularization_loss = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        AE_loss = (loss_part_1 + loss_part_2) / \
            (config.batch_size * config.frame_num) + regularization_loss

        return AE_loss

    @tf.RegisterGradient("CustomSvd")
    def _custom_svd_grad(op, grad_s, grad_u, grad_v):
        """ because there is n gradient for SVD in tensorflow,
        this is our own implementation.
        Define the gradient for SVD, and we just use the eignvalues of the SVD,
        so the gradient formular can be simplified.
        References:
        Ionescu, C., et al, Matrix Backpropagation for Deep Networks with Structured Layers
        Args:
            op: i.e. svd, include the inputs(the input matrix), and the outputs(u,s,v)
            grad_s: the backpropagate gradients from s
            grad_u: the backpropagate gradients from u
            grad_v: the backpropagate gradient from v
        """

        s, u, v = op.outputs
        # Create the shape accordingly.
        u_shape = u.get_shape()[1].value
        v_shape = v.get_shape()[1].value

        eye_mat = tf.eye(u_shape, v_shape)
        # reconstruction the s in matrix format into [batch_size, M, N]
        real_grad_s = tf.matrix_set_diag(eye_mat, grad_s)
        dxdz = tf.matmul(tf.matmul(u, real_grad_s), tf.transpose(v))

        return dxdz

    def DCCAE(self):
        """Deep canonically correlated autoencoders

        TODO: this part need to be checked, and adjust the hyperparamters.
            Besides, ensure the matrix inverse is stable.
        """
        # DCCAE
        # video_input : [batch_size, frame_num, n_input_frame]
        # danmu_input : [batch_size, n_part_danmu, n_input_danmu]
        video_input = tf.reshape(self.data[0], [-1, self.n_input_frame])
        danmu_input = tf.reshape(self.data[1], [-1, self.n_input_danmu])

        self.video_input = video_input
        self.danmu_input = danmu_input
        print('video_input', video_input)
        print('danmu_input', danmu_input)
        video_hidden_size_list = config.DCCAE_video_hidden_size_list
        danmu_hidden_size_list = config.DCCAE_danmu_hidden_size_list

        encoder_video = self.add_layer(
            video_input, video_hidden_size_list[0], video_hidden_size_list[1], 'autoencoder_video_1', activity_func=tf.nn.relu)
        decoder_video = self.add_layer(
            encoder_video, video_hidden_size_list[1], video_hidden_size_list[2], 'autoencoder_video_2', activity_func=tf.nn.relu)

        encoder_danmu = self.add_layer(
            danmu_input, danmu_hidden_size_list[0], danmu_hidden_size_list[1], 'autoencoder_danmu_1', activity_func=tf.nn.relu)
        decoder_danmu = self.add_layer(
            encoder_danmu, danmu_hidden_size_list[1], danmu_hidden_size_list[2], 'autoencoder_danmu_2', activity_func=tf.nn.relu)

        self.encoder_danmu = encoder_danmu
        self.decoder_danmu = decoder_danmu

        self.encoder_video = encoder_video
        self.decoder_video = decoder_video
        # The f(x), g(y) should have the same dimension
        # TODO: Add Noise
        # with tf.variable_scope("video_AE"):
        # initial weights
        # self.initial_AE_weights(name="video_AE", hidden_size = video_hidden_size_list)
        # video_AE_f, video_AE_p_f = self.build_AE(video_input, name = "video_AE", L = len(video_hidden_size_list))
        #
        # # TODO: Add Noise
        # # with tf.variable_scope("danmu_AE"):
        #     # initial weights
        # self.initial_AE_weights(name="danmu_AE", hidden_size = danmu_hidden_size_list)
        # danmu_AE_g, danmu_AE_q_g = self.build_AE(danmu_input, name = "danmu_AE", L = len(danmu_hidden_size_list))
        # tf.get_variable_scope().reuse_variables()
        # DCCAE : CCA part
        # ----f, g is the output we need----------##TODO: CHECK THE OUTPUT
        # video_AE_f is the 2-dim should reshape to 3-dim, [batch_size, frame_num, output_size]

        video_AE_f = encoder_video
        video_AE_p_f = decoder_video

        danmu_AE_g = encoder_danmu
        danmu_AE_q_g = decoder_danmu

        print('video_AE_f', video_AE_f)
        print('danmu_AE_g', danmu_AE_g)

        # video_AE_p_f = decoder_video

        f = video_AE_f
        g = danmu_AE_g

        self.f = f
        self.g = g

        N = config.batch_size * config.frame_num
        F = tf.transpose(f)  # dx*N
        G = tf.transpose(g)
        # remember to ensure F, G are centered
        # F =  self.centered_data(F, config.DCCAE_video_hidden_size_list[1])
        # G = self.centered_data(G, config.DCCAE_danmu_hidden_size_list[1])

        # compute the convariance
        cov11 = tf.matmul(F, tf.transpose(F)) / (N) + \
            config.rx * tf.eye(tf.shape(F)[0])
        cov22 = tf.matmul(G, tf.transpose(G)) / (N) + \
            config.ry * tf.eye(tf.shape(G)[0])
        cov12 = tf.matmul(F, tf.transpose(G)) / (N)

        # cov11 = tf.matmul(F, tf.transpose(F))/(N-1)
        # cov22 = tf.matmul(G, tf.transpose(G))/(N-1)
        # cov12 = tf.matmul(F, tf.transpose(G))/(N-1)

        #tt = self.matrix_square_root(cov22)
        inv_sqrt_cov11 = self.matrix_inv_square_root(cov11)
        inv_sqrt_cov22 = self.matrix_inv_square_root(cov22)
        self.T = tf.matmul(tf.matmul(inv_sqrt_cov11, cov12), inv_sqrt_cov22)

        # SVD decomposition
        # S, U, V = tf.svd(self.T)

        # eps = tf.constant(1e-12, dtype=tf.float32)
        U, V = tf.self_adjoint_eig(tf.matmul(self.T, tf.transpose(self.T)))

        # U, V = tf.self_adjoint_eig(tf.matmul(tf.transpose(self.), cov11))

        # idx = tf.where(tf.greater(U, eps))
        # print('Index', idx, flush=True)
        # # idx = tf.where(idx)
        # print('Index', idx, flush=True)
        # print('U', U)
        # print('V', V)
        # U = tf.gather_nd(U, idx)
        # V = tf.transpose(tf.gather_nd(tf.transpose(V), idx))

        print('U.shape', U, 'V.shape', V)
        print('f.shape', f, 'g.shape', g)
        # self.U = U
        # self.V = V
        # self.S = S

        # self.feat_video = tf.reshape(tf.matmul(f, U), [-1, config.DCCAE_video_hidden_size_list[2]])
        # self.feat_danmu = tf.reshape(tf.matmul(g, V), [-1, config.DCCAE_danmu_hidden_size_list[2]])

        # self.video_feature = tf.matmul(U, )
        # DCCAE loss: CCA part
        # loss_part_1 = -tf.reduce_sum(U)
        val_k, idx_k = tf.nn.top_k(U, k=128)
        loss_part_1 = tf.reduce_sum(val_k)
        # loss_part_1 = 0
        # DCCAE: AutoEncoder part
        # DCCAE loss: AutoEncoder part
        # loss_part_2 = tf.reduce_sum(tf.square(tf.subtract(video_input, video_AE_p_f), 2))*config.DCCAE_lambda
        print('video_input', video_input)
        print('video_AE_p_f', video_AE_p_f)
        loss_part_2 = tf.reduce_sum(tf.square(tf.subtract(
            video_input, video_AE_p_f))) * config.DCCAE_lambda
        loss_part_3 = tf.reduce_sum(tf.square(tf.subtract(
            danmu_input, danmu_AE_q_g))) * config.DCCAE_lambda
        # DCCAE total loss
        DCCAE_loss = (loss_part_1 + loss_part_2 + loss_part_3) / N
        # DCCAE_loss = (loss_part_1)/1

        return DCCAE_loss

    def centered_data(self, inputs, m):
        """centered the data in the DCCAE

        Deep Canonial Correlation Analysis

        Args:
            input: [N, feature_dim]
        Returns:
            norm_input: [N, feature], has been centered and normalized
        """
        print(inputs.shape)
        # m = inputs.get_shape()[1]
        # L = len(config.DCCAE_video_hidden_size_list)
        # m = config.DCCAE_video_hidden_size_list[L//2]
        print('M', m)
        print('Input', inputs)
        tmp_H = tf.matmul(tf.ones(shape=[m, m]), inputs) / m
        print('tmp_H', tmp_H)
        center_input = tf.subtract(inputs, tmp_H)

        return center_input

    def matrix_inv_square_root(self, x):
        """compute the inverse square root of the matix x(real symmetric positive definite)

        Args:
            x : the matirx, should be positive, and the suqare matrix
        Returns:
            sqrtm_x : the matrix square root
        """

        e, v = tf.self_adjoint_eig(x)
        eps = tf.constant(1e-12, dtype=tf.float32)
        U, V = tf.self_adjoint_eig(x)
        idx = tf.greater(e, eps)
        print('Index', idx, flush=True)
        idx = tf.where(idx)
        print('Index', idx, flush=True)
        print('e', e)
        print('v', v)
        e = tf.gather_nd(e, idx)
        v = tf.transpose(tf.gather_nd(tf.transpose(v), idx))

        # sqrtm_x = tf.matmul(tf.matmul(v, tf.diag(1/tf.sqrt(e))), tf.transpose(v))
        sqrtm_x = tf.matmul(
            tf.matmul(v, tf.diag(tf.pow(e, -0.5))), tf.transpose(v))

        return sqrtm_x

    def add_optimize(self):
        """
        Define the optimizer of the model used to train the model
        """
        self.global_step = tf.train.get_or_create_global_step()
        # print('Glo')
        # optimizer = tf.train.AdamOptimizer(self.lr)?
        # optimizer = tf.train.MomentumOptimizer(self.lr, )
        optimizer = tf.train.AdagradOptimizer(self.lr)
        self.optimize = optimizer.minimize(self.loss, self.global_step)

    def add_checkpoint(self):
        """add the checkpoint and other summary information for visualization"""
        # add the saver
        self.saver = tf.train.Saver()
