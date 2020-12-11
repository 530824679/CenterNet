import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from cfg.config import *
from data.dataset import *
from model.network import *

def train():
    dataset_path = path_params['train_data_path']
    log_dir = path_params['logs_path']
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
    inputs, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind = iterator.get_next()

    inputs.set_shape([None, None, None, 3])
    batch_hm.set_shape([None, None, None, None])
    batch_wh.set_shape([None, None, None])
    batch_reg.set_shape([None, None, None])
    batch_reg_mask.set_shape([None, None])
    batch_ind.set_shape([None, None])

    # 构建网络
    model = CenterNet(True)
    pred_hm, pred_wh, pred_reg = model.build_model(inputs)

    # 计算损失
    loss_op = model.calc_loss(pred_hm, pred_wh, pred_reg, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind)

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
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_op[0])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([optimizer, global_step_update]):
                train_op = tf.no_op()
    else:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(solver_params['lr'], global_step, solver_params['decay_steps'], solver_params['decay_rate'], staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss_op[0], global_step=global_step)

    # 配置tensorboard
    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar("hm_loss", loss_op[1])
    tf.summary.scalar("wh_loss", loss_op[2])
    tf.summary.scalar("reg_loss", loss_op[3])
    tf.summary.scalar("total_loss", loss_op[0])
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=60)

    # 模型保存
    save_variable = tf.global_variables()
    saver  = tf.train.Saver(save_variable, max_to_keep=50)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        if solver_params['pre_train']:
            pretrained = np.load(path_params['pretrain_weights'], allow_pickle=True).item()
            for variable in tf.trainable_variables():
                for key in pretrained.keys():
                    key2 = variable.name.rstrip(':0')
                    if (key == key2):
                        sess.run(tf.assign(variable, pretrained[key]))

        summary_writer.add_graph(sess.graph)

        for epoch in range(1, 1 + solver_params['epochs']):
            train_epoch_loss, train_epoch_hm_loss, train_epoch_wh_loss, train_epoch_reg_loss = [], [], [], []
            for index in tqdm(range(batch_num)):
                _, summary, train_total_loss, train_hm_loss, train_wh_loss, train_reg_loss, global_step_val, lr = sess.run([train_op, summary_op, loss_op[0], loss_op[1], loss_op[2], loss_op[3], global_step, learning_rate])

                train_epoch_loss.append(train_total_loss)
                train_epoch_hm_loss.append(train_hm_loss)
                train_epoch_wh_loss.append(train_wh_loss)
                train_epoch_reg_loss.append(train_reg_loss)

                summary_writer.add_summary(summary, global_step_val)

            train_epoch_loss, train_epoch_hm_loss, train_epoch_wh_loss, train_epoch_reg_loss = np.mean(train_epoch_loss), np.mean(train_epoch_hm_loss), np.mean(train_epoch_wh_loss), np.mean(train_epoch_reg_loss)
            print("Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, loss_hm: {:.3f}, loss_wh: {:.3f}, loss_reg: {:.3f}".format(epoch, global_step, lr, train_epoch_loss, train_epoch_hm_loss, train_epoch_wh_loss, train_epoch_reg_loss))
            saver.save(sess, os.path.join(path_params['checkpoints_path'], 'model.ckpt'), global_step=epoch)

        sess.close()

if __name__ == '__main__':
    train()