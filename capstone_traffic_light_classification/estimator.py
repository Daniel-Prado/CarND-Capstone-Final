# -*- coding: utf-8 -*-
from functools import partial
try:
    from .import inception_resnet_v2
    from .inception_preprocessing import preprocess_for_train, preprocess_for_eval
except:
    import inception_resnet_v2
    from inception_preprocessing import preprocess_for_train, preprocess_for_eval


import tensorflow as tf
slim = tf.contrib.slim


class TrainFlags:
    # required
    num_classes = None

    # optional
    master = ''
    train_dir = '/tmp/tfmodel/'
    num_clones = 1
    clone_on_cpu = False
    worker_replicas = 1
    num_ps_tasks = 0
    num_readers = 16
    num_preprocessing_threads = 16
    log_every_n_steps = 10
    save_summaries_secs = 60
    save_interval_secs = 600
    task = 0
    weight_decay = 0.00004
    optimizer = "rmsprop"
    adadelta_rho = 0.95
    adagrad_initial_accumulator_value = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    opt_epsilon = 1.0
    ftrl_learning_rate_power = -0.5
    ftrl_initial_accumulator_value = 0.1
    ftrl_l1 = 0.0
    ftrl_l2 = 0.0
    momentum = 0.9
    rmsprop_decay = 0.9
    learning_rate_decay_type = 'exponential'
    learning_rate = 0.001
    end_learning_rate = 0.00001
    label_smoothing = 0.0
    learning_rate_decay_factor = 0.96
    num_epochs_per_decay = 1.0
    sync_replicas = False
    replicas_to_aggregate = 1
    moving_average_decay = None
    dataset_dir = None
    batch_size = 16
    train_image_size = None
    max_number_of_epochs = None
    checkpoint_path = None
    checkpoint_exclude_scopes = None
    trainable_scopes = None
    ignore_missing_vars = False
    num_samples = 0

    def __init__(
        self,
        num_classes,
        **kwargs
    ):
        self.num_classes = num_classes
        self.__dict__.update(kwargs)


def _get_variables_to_train(FLAGS):  # noqa
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def _configure_optimizer(learning_rate, FLAGS):  # noqa
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _configure_learning_rate(global_step, FLAGS):  # noqa
    """Configures the learning rate.
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    Raises:
      ValueError: if
    """
    decay_steps = int(FLAGS.num_samples / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def model_fn(features, labels, mode, params):
    print(params)
    FLAGS = TrainFlags(**params)
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    images = features["image"]

    arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_resnet_v2.inception_resnet_v2(images, FLAGS.num_classes, is_training=is_training)
        predicts = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        scores = tf.reduce_max(logits, 1)
        predictions = {
            "predictions": predicts,
            "scores": scores
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                "serving_default": tf.estimator.export.PredictOutput(predictions)
            }
        )
    if is_training:
        global_step = tf.train.get_global_step()

        labels = slim.one_hot_encoding(labels, FLAGS.num_classes)
        loss = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=labels,
            label_smoothing=FLAGS.label_smoothing, weights=1.0)
        if 'AuxLogits' in end_points:
            loss += tf.losses.softmax_cross_entropy(
                logits=end_points['AuxLogits'], onehot_labels=labels,
                label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')

        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.histogram('activations/' + end_point, x)
            tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x))
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            tf.summary.scalar('losses/%s' % loss.op.name, loss)

        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        learning_rate = _configure_learning_rate(global_step, FLAGS)
        optimizer = _configure_optimizer(learning_rate, FLAGS)
        tf.summary.scalar('learning_rate', learning_rate)

        if FLAGS.sync_replicas:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            optimizer = tf.train.SyncReplicasOptimizer(
                opt=optimizer,
                replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                variable_averages=variable_averages,
                variables_to_average=moving_average_variables,
                replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
                total_num_replicas=FLAGS.worker_replicas)
        elif FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = _get_variables_to_train(FLAGS)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=learning_rate,
            optimizer=optimizer,
            variables=variables_to_train
        )

        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(labels, logits)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )


def estimator_fn(run_config=None, params={}):
    FLAGS = TrainFlags(**params)
    return tf.estimator.Estimator(model_fn, model_dir=FLAGS.train_dir, params=params, config=run_config)


def train_input_fn(training_dir, params):
    preprocess_fn = partial(preprocess_for_train, bbox=None, random_flip=True, random_resize=True)
    with tf.name_scope("train_input"):
        return _input_fn(preprocess_fn, training_dir, params)


def eval_input_fn(training_dir, params):
    with tf.name_scope("eval_input"):
        return _input_fn(preprocess_for_eval, training_dir, params)


def _input_fn(preprocess_fn, training_dir, params):
    FLAGS = TrainFlags(**params)

    batch_size = FLAGS.batch_size
    train_image_size = int(FLAGS.train_image_size or inception_resnet_v2.inception_resnet_v2.default_image_size)

    def parse_record(record):
        features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature((), tf.string, default_value=''),
        }
        parsed = tf.parse_single_example(record, features)
        decoded_image = tf.image.decode_image(parsed["image/encoded"], channels=3)
        decoded_image.set_shape([None, None, 3])
        image = preprocess_fn(decoded_image, train_image_size, train_image_size)
        return {"image": image}, parsed["image/class/label"]

    # file_pattern = os.path.join(training_dir, "*.tfrecord")
    if "*" in training_dir:
        dataset = tf.data.Dataset.list_files(training_dir)
    elif isinstance(training_dir, str):
        training_dir = [training_dir]
    dataset = tf.data.Dataset.from_tensor_slices(training_dir)

    dataset = dataset.interleave(
        lambda file_name: (
            tf.data.TFRecordDataset(file_name)
            .map(parse_record)
            .batch(batch_size)
            .repeat(FLAGS.max_number_of_epochs)
        ), cycle_length=8
    )
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


# def serving_input_fn(params):
#     pass
