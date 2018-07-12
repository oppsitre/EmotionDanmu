# TODO: check all the models and implementations, and add their hyper parameter in this file. Benifit for fine tuning.
save_model = True  # if true, then the graph model information will be savedevry epoch once
add_noise = False
save_model_step = 1000
# if true, the we will read the data in sequence, and save the feature to disk. TODO
unsup_save_data = False
lr = 1e-2  # learning rate
# vector_size = 128 #the size of word2vector
n_part_danmu = 10  # the lenght of lstm of danmu
epoch_size = 10000000  # the number of training epoch
batch_size = 2048  # the number of input in each batch
modality = [0, 1]  # choose which modalities will be used ['video', 'text']
ratio_split = [0.8, 0.2]  # ratio_split = [train_ration, test_ratio]
n_hidden_comment = 1024  # the size of hidden layer in modality of comment
n_hidden_danmu = 4096  # the size of hidden layer in modality of danmu
n_hidden_video = 4096  # the size of hidden layer in modality of video
n_hidden_classification = 2048 * 2  # the size of hidden layer of classification
# the size of hidden layer of classification
n_hidden_classification_1 = 2048 * 2
# the size of hidden layer of classificatio1
n_hidden_classification_2 = 2048 * 2
# n_hidden_classification_3 = 8096 #the size of hidden layer of classification/
n_hidden_danmu_feature = 512
n_hidden_comment_feature = 512
# the list of output feature of each modality [comment, danmu]'
n_output_modality = [2048, 2048]
n_hidden_lstm_video = 512
n_hidden_lstm_danmu = 512
optimize = 'adam'  # the method of optimization
loss = 'rmse'  # the function of loss
test_step = 100  # step of print the loss of test set
# topic_number = 10 # the number of topics
emotion_number = 7  # the number of emotions
topic_vector_size = 1
word_vector_size = 256
emotion_vector_size = 1
file_doc_embedding = '../dataset/texts/doc_embedding_tw1_we_7.txt'
file_doc_id = '../dataset/texts/doc2id.txt'
file_wordmap = '../dataset/texts/wordmap.txt'
# file_topic_emotion_assign = '../../JST_py/data/final.tassign'
folder_save = '../result_dir/temporal_flows'
train_ratio = 0.2
file_data_train = '../dataset/texts/video_class_train.txt'
# file_data_train = '../dataset/texts/video_class_train_0.2.txt'
file_data_test = '../dataset/texts/video_class_test.txt'
# file_data_train = '../dataset/texts/video_class_train_jinchen.txt'
# file_data_test = '../dataset/texts/video_class_test_jinchen.txt'
doc_vector_size = 0  # this will be read from the file
n_classes = 7  # the number of classes will be predicted
lamda = 0.1  # hyperparameter
train_status = 1  # 0: autoencoder pretrain, 1: supervised training
frame_num = 10
n_input_frame = 4096  # refer to fc_sizen
frame_per_cluster = 3
print_step = 20
log = "log"
n_input_danmu = doc_vector_size
file_feature_video = 'features_video.npy'
file_feature_danmu = 'features_danmu.npy'
unsupervised_network = 'DCCAE'  # ['DCCAE', 'DistAE', no,]
#########----------Emotion LDA-----------##
topic_number = 7
lda_alpha = 50 / topic_number
lda_beta = 0.1
lda_gamma = 0.3
#######----------AutoEncoder Part---------##############
###########----1.DistAE-----#############
DistAE_lambda = 1e-3  # hyper-parameter
DistAE_video_hidden_size_list = [
    n_input_frame] + [4096, 256, 4096] + [n_input_frame]
DistAE_danmu_hidden_size_list = [
    n_input_danmu] + [8196, 256, 8196] + [n_input_danmu]
# --------------------#########------------------------
rx = 1  # hyper paramter DCCAE: the regularization in the loss function for the video #1E-4
ry = 1  # hyper parameter DCCAE: the regularization in the loss function for the danmu
DCCAE_lambda = 1e-2  # hyper parameter  bigger , the reconstruction more important
DCCAE_video_hidden_size_list = [n_input_frame] + [256] + [n_input_frame]
DCCAE_danmu_hidden_size_list = [n_input_danmu] + [256] + [n_input_danmu]
#########----------################################################
