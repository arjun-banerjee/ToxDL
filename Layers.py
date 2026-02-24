__author__ = 'jasper.zuallaert'
import tensorflow as tf

# Dynamic max pooling layer
# Relatively dirtily programmed, as I did not find another way to achieve this pooling effect
# Function parameters:
# - input: tf Tensor of shape (n, x, n_filters) with n the number of samples, the maximum sequence length after the
#          previous pooling operations, and n_filters the amount of filters as a result of the last convolutional layer
# - input_lengths: tf Tensor of shape (n) with the actual lengths of each input sample
# - fixed_output_size: the dynamic max pooling size, indicating how many outputs the maxpooling should produce in the end
#                      e.g., for input of (64, 100, 200) and dynamic maxpool size of 10, the output shape will be
#                      (64, 10, 200)
def dynamic_max_pooling_with_overlapping_windows(input, input_lengths, fixed_output_size): #with half overlapping pooling windows
    y = 0.5*(fixed_output_size+1)
    part = tf.math.ceil(tf.cast(input_lengths,dtype=tf.float32)/y)
    w = input.shape[1]
    p = input.shape[2]

    assert len(input.shape) == 3, 'input should have three axes'

    reshaped_input = tf.transpose(input,perm=[0,2,1])
    allVecs = []
    for i in range(fixed_output_size):
        mask = tf.sequence_mask(part * 0.5*(i+2),maxlen=w,dtype=tf.float32) - (tf.sequence_mask(part*0.5*(i),maxlen=w,dtype=tf.float32) if i > 0 else 0)
        mask = tf.expand_dims(mask,1)
        mask_all_axes = tf.tile(mask,[1,p,1])
        calc = mask_all_axes * reshaped_input
        pooled = tf.reduce_max(calc,axis=2,keepdims=True)
        allVecs.append(pooled)

    concatenated = tf.transpose(tf.concat(allVecs,axis=2),perm=[0,2,1])
    return concatenated


# Bidirectional GRU layer using Keras layers (TF2 compatible)
# Function parameters:
# - input: tf Tensor of shape (n, x, n_filters) with n the number of samples, the maximum sequence length after the
#          previous pooling operations, and n_filters the amount of filters as a result of the last convolutional layer
# - input_lengths: tf Tensor of shape (n) with the actual lengths of each input sample (so the GRU knows where to stop)
# - state_size: the hidden state size for the GRUs
def BidirectionalGRULayer(input, input_lengths, state_size):
    # Create forward and backward GRU layers
    forward_gru = tf.keras.layers.GRU(state_size, return_state=True)
    backward_gru = tf.keras.layers.GRU(state_size, return_state=True, go_backwards=True)

    # Run forward and backward passes
    _, forward_state = forward_gru(input)
    _, backward_state = backward_gru(input)

    # Concatenate the final states
    lastCombined = tf.concat([forward_state, backward_state], axis=1)
    return lastCombined


## Focal loss function, copied from the WWW, as a very quick experiment
def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = tf.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = tf.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


# A helper function to print out the first trainable layer. This layer can be two things:
# - if trainable embeddings are chosen, the embeddings for all amino acids are printed out (len(values.shape) == 2)
# - if no trainable embeddings are chosen, the first convolutional filters are printed out (len(values.shape) == 3)
#           note: only trainable embeddings tested - there should be a double-check to make sure
#           that convolutional filters are properly printed, and not for instance flipped / reversed
# Function parameters:
# - saveToDir: the directory where the tensorflow model was saved (should be within the parameters/ directory)
def visualizeFirstTrainableLayer(saveToDir):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Try to load as Keras model first
    try:
        model = tf.keras.models.load_model(saveToDir)
        # Get the first trainable layer's weights
        for layer in model.layers:
            if layer.trainable_weights:
                values = layer.trainable_weights[0].numpy()
                break
    except:
        # Fall back to checkpoint loading for legacy models
        filename = saveToDir+'/'+saveToDir[saveToDir.rfind('/')+1:]
        try:
            checkpoint = tf.train.load_checkpoint(filename)
            # Try to get the first variable (embedding or conv weights)
            var_names = [name for name, _ in tf.train.list_variables(filename)]
            # Find embedding or first trainable variable
            for var_name in var_names:
                if 'embedding' in var_name.lower() or 'Variable' in var_name:
                    values = checkpoint.get_tensor(var_name)
                    break
            else:
                values = checkpoint.get_tensor(var_names[0])
        except Exception as e:
            print(f'Could not load checkpoint: {e}')
            return

    if len(values.shape) == 3:
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        for filterN in range(len(values[0][0])):
            print(f'### FILTER {filterN:03d} ###')
            for aaN in range(len(values[0])):
                print(f'{amino_acids[aaN]}  ',end='')
                for pos in range(len(values)):
                    print(f'{values[pos][aaN][filterN]: 1.2f}', end=' ')
                print()
            print('###################')
            print()
    elif len(values.shape) == 2:
        amino_acids = '_ACDEFGHIKLMNPQRSTVWY'
        for aaN in range(len(values)):
            print(f'{amino_acids[aaN]}  ', end='')
            for dim in range(len(values[0])):
                print(f'{values[aaN][dim]: 1.2f}', end=' ')
            print()
    else:
        print(f'No support for values of shape: {values.shape}')


import sys
if sys.argv[0] == 'Layers.py':
    if len(sys.argv) != 2:
        print('Usage: python Layers.py <test_218390_129039>')
    else:
        visualizeFirstTrainableLayer(sys.argv[1])
