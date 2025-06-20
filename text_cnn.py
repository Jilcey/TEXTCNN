import tensorflow as tf


def TextCNN(sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
            l2_reg_lambda=0.0, embedding_matrix=None, trainable=True, multichannel=False,
            use_attention=False, use_dilated=False, dilation_rate=1):
    """Implement 4 model variants with optional dilated convolution and attention."""
    tf.compat.v1.disable_eager_execution()

    input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="input_x")
    input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")
    dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

    with tf.compat.v1.variable_scope("embedding"):
        if multichannel:
            with tf.compat.v1.variable_scope("static"):
                embedding_static = tf.compat.v1.get_variable(
                    "W", initializer=tf.constant(embedding_matrix[0], dtype=tf.float32), trainable=False
                )
                embedded_static = tf.nn.embedding_lookup(embedding_static, input_x)
            with tf.compat.v1.variable_scope("non_static"):
                embedding_non_static = tf.compat.v1.get_variable(
                    "W", initializer=tf.constant(embedding_matrix[1], dtype=tf.float32), trainable=trainable
                )
                embedded_non_static = tf.nn.embedding_lookup(embedding_non_static, input_x)
            embedded_chars = tf.stack([embedded_static, embedded_non_static], axis=-1)
        else:
            embedding = tf.compat.v1.get_variable(
                "W", shape=[vocab_size, embedding_size],
                initializer=tf.constant_initializer(
                    embedding_matrix) if embedding_matrix is not None else tf.compat.v1.random_uniform_initializer(-0.1, 0.1),
                trainable=trainable
            )
            embedded_chars = tf.nn.embedding_lookup(embedding, input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    pooled_outputs = []
    for filter_size in filter_sizes:
        with tf.compat.v1.variable_scope(f"conv-maxpool-{filter_size}"):
            if multichannel:
                filter_shape = [filter_size, embedding_size, 2, num_filters]
                conv_input = embedded_chars
            else:
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                conv_input = embedded_chars_expanded

            W = tf.compat.v1.get_variable(
                "W", shape=filter_shape,
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
                regularizer=tf.compat.v1.nn.l2_loss if l2_reg_lambda > 0 else None
            )
            b = tf.compat.v1.get_variable(
                "b", shape=[num_filters], initializer=tf.compat.v1.constant_initializer(0.1)
            )

            if use_dilated:
                dilations = [1, dilation_rate, 1, 1]
                conv = tf.nn.conv2d(
                    input=conv_input, filters=W, strides=[1, 1, 1, 1], padding="VALID", dilations=dilations
                )
                conv_output_height = sequence_length - (filter_size - 1) * dilation_rate
            else:
                conv = tf.nn.conv2d(
                    input=conv_input, filters=W, strides=[1, 1, 1, 1], padding="VALID"
                )
                conv_output_height = sequence_length - filter_size + 1

            assert conv_output_height > 0, \
                f"Convolution output height must be >0 (seq_len={sequence_length}, filter={filter_size}, dilation={dilation_rate})"

            h = tf.nn.tanh(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool2d(
                input=h,
                ksize=[1, conv_output_height, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID"
            )
            pooled = tf.reshape(pooled, [-1, num_filters])
            pooled_outputs.append(pooled)

    h_pool = tf.concat(pooled_outputs, axis=1)
    h_drop = tf.nn.dropout(h_pool, rate=1.0 - dropout_keep_prob)

    if use_attention:
        attention_size = h_pool.shape[1]
        with tf.compat.v1.variable_scope("attention"):
            W_att = tf.compat.v1.get_variable(
                "W", shape=[attention_size, attention_size],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
                regularizer=tf.compat.v1.nn.l2_loss if l2_reg_lambda > 0 else None
            )
            b_att = tf.compat.v1.get_variable("b", shape=[attention_size],
                                              initializer=tf.compat.v1.constant_initializer(0.1))
            u_att = tf.compat.v1.get_variable(
                "u", shape=[attention_size],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
                regularizer=tf.compat.v1.nn.l2_loss if l2_reg_lambda > 0 else None
            )
            h_att = tf.tanh(tf.matmul(h_pool, W_att) + b_att)
            att_logits = tf.matmul(h_att, tf.expand_dims(u_att, -1))
            alpha = tf.nn.softmax(att_logits, axis=1)
            h_att = h_pool * alpha
            h_drop = tf.nn.dropout(h_att, rate=1.0 - dropout_keep_prob)
    else:
        h_drop = tf.nn.dropout(h_pool, rate=1.0 - dropout_keep_prob)

    with tf.compat.v1.variable_scope("output"):
        num_total_features = num_filters * len(filter_sizes)
        W = tf.compat.v1.get_variable(
            "W", shape=[num_total_features, num_classes],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
            regularizer=tf.compat.v1.nn.l2_loss if l2_reg_lambda > 0 else None
        )
        b = tf.compat.v1.get_variable("b", shape=[num_classes], initializer=tf.compat.v1.constant_initializer(0.1))
        logits = tf.matmul(h_drop, W) + b
        predictions = tf.nn.softmax(logits, name="predictions")

    with tf.compat.v1.variable_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits)
        loss = tf.reduce_mean(losses)
        if l2_reg_lambda > 0:
            l2_loss = tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
            loss += l2_reg_lambda * l2_loss

    with tf.compat.v1.variable_scope("accuracy"):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return input_x, input_y, dropout_keep_prob, predictions, loss, accuracy
