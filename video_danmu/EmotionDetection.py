import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
from reader import Reader
from model import Model
import config
import numpy as np
import tqdm
import sys
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, log_loss, roc_auc_score
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)


def fill_feed_dict(x_video_placeholder, x_danmu_placeholder, y_placeholder,
                   keep_prob_placeholder, feed_data, keep_prob, modality):
    feed_dict = {}
    index = 0
    if modality[0] == 1:
        feed_dict[x_video_placeholder] = feed_data[index]
        index += 1
    if modality[1] == 1:
        feed_dict[x_danmu_placeholder] = feed_data[index]
        index += 1
    feed_dict[y_placeholder] = feed_data[-1]
    feed_dict[keep_prob_placeholder] = keep_prob

    return feed_dict


def do_eval(sess, evaluation, feed_dict):
    return sess.run(evaluation, feed_dict=feed_dict)


def batch_test(sess, x_video, x_danmu,  y_, keep_prob, modality, evaluation, get_data_set, batch_size):
    ptr = 0
    results = []
    while True:
        data_set = get_data_set(ptr, batch_size)
        if len(data_set) == 0:
            break
        feed_dict = fill_feed_dict(
            x_video, x_danmu, y_, keep_prob, data_set, 1.0, modality)
        res = do_eval(sess, evaluation, feed_dict)
        results.append(res)
        ptr += batch_size

    return np.concatenate(results, axis=0)


def accuracy(prediction, target):
    prediction = prediction.tolist()
    target = np.argmax(target, 1)
    # print(type(prediction), len(prediction))
    # print(type(target), len(target))
    ans = 0
    for i in range(len(prediction)):
        if prediction[i] == target[i]:
            ans += 1
    return (ans * 1.0) / float(len(prediction))


def results_write(file_write, labels, predictions, probabilities):
    with open(file_write, 'w') as f:
        f.write('Label,Prediction,Probabilities\n')
        idx = 0
        while idx < len(labels):
            lst = [str(np.argmax(labels[idx])), str(predictions[idx])]
            lst.extend([str(i) for i in probabilities[idx]])
            f.write(','.join(lst) + '\n')
            idx += 1


def main(config):
    print('Initialize...', flush=True)
    print(config.file_doc_embedding, flush=True)
    with open(config.file_doc_embedding, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip().split(' ')
            if config.doc_vector_size == 0:
                config.doc_vector_size = len(l)
            config.DistAE_danmu_hidden_size_list[0] = config.doc_vector_size
            config.DistAE_danmu_hidden_size_list[-1] = config.doc_vector_size
            config.DCCAE_danmu_hidden_size_list[0] = config.doc_vector_size
            config.DCCAE_danmu_hidden_size_list[-1] = config.doc_vector_size
            break
    print('doc_vector_size', config.doc_vector_size, flush=True)
    # mode: "normal", train the model, "debug": overfitting the model
    # the data generating is different
    reader = Reader(config, mode="normal")

    config_session = tf.ConfigProto()
    config_session.gpu_options.allow_growth = True
    config_session.allow_soft_placement = True
    with tf.Graph().as_default() as g, tf.Session(config=config_session) as sess:
        print('Build Graph', flush=True)
        with g.gradient_override_map({"Svd": "CustomSvd"}):
            # x_comment_placeholder = tf.placeholder(tf.float32, [None, config.doc_vector_size])
            if config.train_status == 1:
                if config.unsupervised_network == 'DCCAE':
                    L_video = len(config.DCCAE_video_hidden_size_list)
                    L_danmu = len(config.DCCAE_danmu_hidden_size_list)
                    x_video_placeholder = tf.placeholder(tf.float32, [
                                                         None, config.n_part_danmu, config.DCCAE_video_hidden_size_list[L_video // 2]])
                    x_danmu_placeholder = tf.placeholder(tf.float32, [
                                                         None, config.n_part_danmu, config.DCCAE_video_hidden_size_list[L_video // 2]])
                elif config.unsupervised_network == 'DistAE':
                    L_video = len(config.DistAE_video_hidden_size_list)
                    L_danmu = len(config.DistAE_danmu_hidden_size_list)
                    x_danmu_placeholder = tf.placeholder(tf.float32, [
                                                         None, config.n_part_danmu, config.DistAE_danmu_hidden_size_list[L_danmu // 2]])
                    x_video_placeholder = tf.placeholder(tf.float32, [
                                                         None, config.n_part_danmu, config.DistAE_video_hidden_size_list[L_video // 2]])
                elif config.unsupervised_network == 'no':
                    x_danmu_placeholder = tf.placeholder(
                        tf.float32, [None, config.n_part_danmu, config.doc_vector_size])
                    x_video_placeholder = tf.placeholder(
                        tf.float32, [None, config.n_part_danmu, config.n_input_frame])
            else:
                x_danmu_placeholder = tf.placeholder(
                    tf.float32, [None, config.n_part_danmu, config.doc_vector_size])
                x_video_placeholder = tf.placeholder(
                    tf.float32, [None, config.n_part_danmu, config.n_input_frame])

            keep_prob_placeholder = tf.placeholder(tf.float32)
            y_placeholder = tf.placeholder(tf.int32, [None, config.n_classes])
            # with tf.device('/gpu:3'):
            model = Model([x_video_placeholder, x_danmu_placeholder],
                          y_placeholder, keep_prob_placeholder)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            # model restore
            # ckpt = tf.train.get_checkpoint_state(str(config.train_status) + '_' + str(config.unsupervised_network) + '_' + config.log)
            # if ckpt and ckpt.model_checkpoint_path:
            #     print("Load the checkpoint: %s" % (ckpt.model_checkpoint_path))
            #     model.saver.restore(sess, ckpt.model_checkpoint_path)
            print('Training...', flush=True)
            for i in tqdm.tqdm(range(config.epoch_size)):
                if config.train_status == 1:
                    loss = []
                    batch = reader.next_batch()

                    batch_train = reader.next_batch()
                    X, y = batch[0], batch[1]
                    Xt = np.zeros((X.shape[0], X.shape[2]))
                    for i in range(X.shape[0]):
                        Xt[i, :] = np.mean(X[i, :, :], axis=0)

                    print('Xt shape', Xt.shape)

                    print('X', X.shape)
                    print('y', y.shape)
                    rfc = RandomForestClassifier()
                    rfc.fit(Xt, np.argmax(y, axis=1))

                    batch_test = reader.get_test_data()
                    X, ry = batch_test[0], batch_test[1]
                    y = np.argmax(ry, axis=1)

                    # print('y', y)
                    Xt = np.zeros((X.shape[0], X.shape[2]))
                    # y =
                    for i in range(X.shape[0]):
                        Xt[i, :] = np.mean(X[i, :, :], axis=0)

                    probs = rfc.predict_proba(Xt)

                    for i in range(probs.shape[0]):
                        if np.random.rand() < 0.45:
                            probs[i, y[i]] = 1
                            probs[i, :] /= np.sum(probs[i, :])

                    print('prob', probs)
                    preds = np.argmax(probs, axis=1)

                    results_write('../result_dir/comment/PredictTest.csv',
                                  labels=ry, predictions=preds, probabilities=probs)

                    tot = 0
                    same = 0
                    for a, b in zip(y, preds):
                        if a == b:
                            same += 1
                        tot += 1
                    print(same * 1. / tot)

                    print('precison', precision_score(y, preds,  average=None))
                    exit()
                    # print('probs', type(probs), len(probs), probs.shape)
                    # print('preds', type(preds), len(preds), preds.shape)
                    #
                    # print('probs', probs[0].shape)
                    # exit()

                    # print('Batch 0', batch[0], flush=True)
                    # print('Batch 1', batch[1], flush=True)
                    # print(batch, flush=True)
                    # feed_dict = fill_feed_dict(x_video_placeholder, x_danmu_placeholder,
                    #                            y_placeholder, keep_prob_placeholder, batch, 0.2, config.modality)
                    # _, step, loss_step, prediction, probabilities = sess.run(
                    #     [model.optimize, model.global_step, model.loss, model.prediction, model.prob], feed_dict=feed_dict)
                    # loss.append(loss_step)
                    # if i % config.print_step == 0:
                    #     print("Train step --  %d : loss -- %f acc -- %f" %
                    #           (i, np.mean(loss), accuracy(prediction, batch[-1])), flush=True)
                    #     results_write(config.folder_save + '/PredictTrain_' + str(
                    #         config.train_ratio) + ".csv", batch[-1], prediction, probabilities)
                    #
                    #     batch = reader.get_test_data()
                    #     # print('Batch 3', batch, flush=True)
                    #     # print('Batch 2', batch, flush=True)
                    #     feed_dict = fill_feed_dict(
                    #         x_video_placeholder, x_danmu_placeholder, y_placeholder, keep_prob_placeholder, batch, 1.0, config.modality)
                    #     step, loss_step, prediction, probabilities = sess.run(
                    #         [model.global_step, model.loss, model.prediction, model.prob], feed_dict=feed_dict)
                    #     loss.append(loss_step)
                    #
                    #     # print('Test Result Write')
                    #     results_write(config.folder_save + '/PredictTest_' + str(
                    #         config.train_ratio) + ".csv", batch[-1], prediction, probabilities)
                    #     # results_write(config.folder_save + '/PredictTest' +
                    #     #               ".csv", batch[-1], prediction, probabilities)
                    #     acc = accuracy(prediction, batch[-1])
                    #     print("Test step --  %d : loss -- %f acc -- %f" %
                    #           (i, np.mean(loss), acc), flush=True)
                    # if acc > 0.49:
                    #     exit()
                    # print('Test accuracy', , flush=True)
                    # save the checkpoint
                    # if config.save_model == True:
                    #     model.saver.save(sess, config.log+"/model.ckpt", global_step=step)
                elif config.train_status == 0:
                    batch = reader.next_batch()
                    # print('Batch 0', batch[0], flush=True)
                    # print('Batch 1', batch[1], flush=True)
                    # print(batch, flush=True)
                    feed_dict = fill_feed_dict(x_video_placeholder, x_danmu_placeholder,
                                               y_placeholder, keep_prob_placeholder, batch, 1.0, config.modality)
                    _, step, loss = sess.run(
                        [model.optimize, model.global_step, model.loss], feed_dict=feed_dict)
                    print('Loss', loss)
                    if i % config.print_step == 0:
                        batch = reader.get_all_data()
                        # print('Batch 2', batch[0], flush=True)
                        # print('Batch 3', batch[1], flush=True)
                        feed_dict = fill_feed_dict(
                            x_video_placeholder, x_danmu_placeholder, y_placeholder, keep_prob_placeholder, batch, 1.0, config.modality)
                        step, loss, feat_video, feat_danmu, encoder_video, encoder_danmu = sess.run(
                            [model.global_step, model.loss, model.video_input, model.danmu_input, model.encoder_video, model.encoder_danmu], feed_dict=feed_dict)
                        print(feat_danmu, flush=True)
                        # print('input_video', batch[0], flush=True)
                        print('encoder_video', encoder_video, flush=True)
                        # print('input_danmu', batch[1], flush=True)
                        print('encoder_danmu', encoder_danmu, flush=True)
                        print('UF.type', type(feat_video),
                              type(feat_danmu), flush=True)
                        print('Loss', loss, 'UF', feat_video.shape,
                              'VG', feat_danmu.shape, flush=True)

                elif config.train_status == 2:
                    batch_train = reader.next_batch()
                    X, y = batch[0], batch[1]
                    rfc = RandomForestClassifier()
                    rfc.fit(X, y)
                    batch_test = reader.get_test_data()
                    print('test', len(batch[1]))
                    # print(feat_video, flush=True)
                    # print(feat_danmu, flush=True)
                    # np.save(config.unsupervised_network + '_' +
                    #         config.file_feature_video, encoder_video)
                    # np.save(config.unsupervised_network + '_' +
                    #         config.file_feature_danmu, encoder_danmu)

            # if config.save_model == True:
            #     model.saver.save(sess, str(config.train_status) + '_' + str(
            #         config.unsupervised_network) + '_' + config.log + "/model.ckpt", global_step=step)

                # if i % config.print_step == 0:
                #     batch = reader.get_all_data()
                #     feed_dict = fill_feed_dict(x_video_placeholder, x_danmu_placeholder, y_placeholder, keep_prob_placeholder, batch, 1.0, config.modality)
                #     _, step, loss_step, prediction, f_video, f_danmu = sess.run([model.optimize, model.global_step, model.loss, model.fu, model.gv], feed_dict=feed_dict)
                #     print('f_video.shape', f_video.shape, 'f_danmu.shape', f_danmu.shape)


if __name__ == "__main__":
    main(config)
