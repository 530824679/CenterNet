import os
import numpy as np
import tensorflow as tf
from cfg.config import *
from data.dataset import *

def train():
    dataset_path = path_params['train_data_path']
    log_dir = path_params['logs_dir']
    batch_size = solver_params['batch_size']
    lr_type = solver_params['lr_type']

    # 配置GPU
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # 解析得到训练样本以及标注
    image_num = len(open(dataset_path, 'r').readlines())
    batch_num = int(math.ceil(float(image_num) / batch_size))
    dataset = create_dataset(dataset_path, batch_num, batch_size=batch_size, is_shuffle=True)
    iterator = dataset.make_one_shot_iterator()
    inputs, y_true_13, y_true_26, y_true_52 = iterator.get_next()

    # 构建网络
    model =

    # 计算损失

    # 定义优化方式
    if lr_type == "CosineAnnealing":
        global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
        warmup_steps = tf.constant(solver_params['warm_up_epochs'] * batch_num, dtype=tf.float64, name='warmup_steps')
        train_steps = tf.constant(solver_params['epochs'] * batch_num, dtype=tf.float64, name='train_steps')
        learning_rate = tf.cond(pred=global_step < warmup_steps,
            true_fn=lambda: global_step / warmup_steps * solver_params['init_lr'],
            false_fn=lambda: solver_params['end_lr'] + 0.5 * (solver_params['init_lr'] - solver_params['end_lr']) *
                             (1 + tf.cos((global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
        )
        global_step_update = tf.assign_add(global_step, 1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([optimizer, global_step_update]):
                train_op = tf.no_op()
    else:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(solver_params['lr'], global_step, solver_params['lr_decay_steps'], solver_params['lr_decay_rate'], staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=global_step)

    # 配置tensorboard
    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar("hm_loss", hm_loss)
    tf.summary.scalar("wh_loss", wh_loss)
    tf.summary.scalar("reg_loss", reg_loss)
    tf.summary.scalar("total_loss", total_loss)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=60)

    # 模型保存
    save_variable = tf.global_variables()
    saver  = tf.train.Saver(save_variable, max_to_keep=50)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        if solver_params['pre_train']:
            load_weights(sess, path_params['pretrain_weights'])

        summary_writer.add_graph(sess.graph)

        for epoch in range(1, 1 + solver_params['total_epoches']):
            pbar = tqdm(range(num_train_batch))
            train_epoch_loss, test_epoch_loss = [], []
            sess.run(trainset_init_op)
            for i in pbar:
                _, summary, train_step_loss, global_step_val = sess.run(
                    [train_op, write_op, total_loss, global_step], feed_dict={is_training: True})

                train_epoch_loss.append(train_step_loss)
                summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" % train_step_loss)

            sess.run(testset_init_op)
            for j in range(num_test_batch):
                test_step_loss = sess.run(total_loss, feed_dict={is_training: False})
                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./checkpoint/centernet_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            saver.save(sess, ckpt_file, global_step=epoch)

if __name__ == '__main__':
    train()