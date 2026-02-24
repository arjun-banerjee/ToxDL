__author__ = 'jasper.zuallaert'

import numpy as np
import GO_Graph_Builder as gogb
import tensorflow as tf

from Layers import dynamic_max_pooling_with_overlapping_windows, BidirectionalGRULayer

# Returns a NetworkObject object, containing the neural network created, and the placeholders that come with it
# The network topology can be constructed with a broad variation of parameters, explained below:
# - type: The topology type ('G' = DeepGO, 'D' = dynamic max pooling, 'K' = k-max pooling, 'O' = zero-padding only,
#                            'R' = GRU, 'M' = single max pooling, 'C' = combined D+K, 'P' = domain vector network only)
# - maxLength: the maximum sequence length, which should be the same as the one specified when reading the Datasets in
#              InputManager.py
# - ngramsize
# - filterSizes: list with the filter size of each convolutional layer                      e.g. [9,7,7]
# - filterAmounts: list with the amount of filters of each convolutional layer              e.g. [100,200,300]
# - maxPoolSizes: list with the max pool sizes of each max pooling layer                    e.g. [2,2,1]
#   Note: filterSizes, filterAmounts and maxPoolSizes should all have the same length
# - sizeOfFCLayers: an integer indicating the size of the fully-connected layers at the end e.g. 64
# - n_of_outputs: the amount of classes for which a prediction should be done by the network
# - dynMaxPoolSize: in case of dynamic or k-max pooling, this integer indicates the amount of output 'buckets'
# - term_indices_file: the mapping file between class indices and GO terms (e.g. inputs/mf.indices)
# - ppi_vectors: a boolean, indicating whether or not domain vectors should be included in the network
# - hierarchy: a boolean, indicating whether or not the hierarchy strategy as described in DeepGO should be added
#              at the output layers
# - embeddingType: the type of embedding used, should be one of 'trainable' or 'onehot'
# - embeddingDepth: in case of trainable embeddings, how large the vector should be is indicated by this integer
# - GRU_state_size: in case of a GRU network, what the size of the hidden state should be is indicated by this integer
def buildNetworkTopology(type,
                         maxLength,
                         ngramsize,
                         filterSizes,
                         filterAmounts,
                         maxPoolSizes,
                         sizeOfFCLayers,
                         n_of_outputs,
                         dynMaxPoolSize,
                         term_indices_file,
                         ppi_vectors,
                         hierarchy,
                         embeddingType,
                         embeddingDepth,
                         GRU_state_size):
    maxLength = maxLength - ngramsize + 1

    if type == 'G':
        return buildDeepGO(ngramsize, n_of_outputs, term_indices_file, ppi_vectors, hierarchy, maxLength)
    elif type in 'DKORMC':
        return buildMyNetwork(type, ngramsize, n_of_outputs, term_indices_file, filterSizes, filterAmounts, maxPoolSizes, sizeOfFCLayers,
                   dynMaxPoolSize, ppi_vectors, hierarchy, embeddingDepth, embeddingType, GRU_state_size, maxLength)
    elif type == 'P':
        return buildPPIOnlyNetwork(n_of_outputs, term_indices_file, sizeOfFCLayers, hierarchy, maxLength)
    else:
        return AssertionError('Type {} not supported'.format(type))


#######################################################################################################
#######################################################################################################
#######################################################################################################

# Prints the details of the neural network (layers and output shapes), except for the output layers
def printNeuralNet(layers):
    print('Network information:')
    for l in layers:
        try:
            print('{:35s} -> {}'.format(l.name,l.shape))
        except AttributeError:
            pass

#######################################################################################################
#######################################################################################################
#######################################################################################################

# Returns the DeepGO model. Explanation of the parameters can be found on top of this file
def buildDeepGO(ngramsize, n_of_outputs, term_indices_file, ppi_vectors, hierarchy, maxLength):
    assert ngramsize == 3

    # Define inputs
    X_input = tf.keras.Input(shape=(maxLength,), dtype=tf.int32, name='X_placeholder')
    seqlen_input = tf.keras.Input(shape=(), dtype=tf.int32, name='seqlen_placeholder')
    vec_input = tf.keras.Input(shape=(256,), dtype=tf.float32, name='vec_placeholder')

    # Embedding layer
    vocab_size = 20**ngramsize + 1
    embedding_init = tf.keras.initializers.RandomUniform(-0.05, 0.05)
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=128,
        embeddings_initializer=embedding_init,
        trainable=True,
        mask_zero=True,
        name='embedding_out'
    )

    layers = []
    l = embedding_layer(X_input)
    layers.append(l)
    layers.append(tf.keras.layers.Dropout(0.2)(layers[-1]))

    layers.append(tf.keras.layers.Conv1D(32, 128, padding='valid', activation='relu')(layers[-1]))
    layers.append(tf.keras.layers.MaxPooling1D(64, 32)(layers[-1]))
    layers.append(tf.keras.layers.Flatten()(layers[-1]))

    if ppi_vectors:
        layers.append(tf.keras.layers.Concatenate()([layers[-1], vec_input]))

    logits = []
    output_layers = [None] * n_of_outputs
    if not hierarchy:
        for i in range(n_of_outputs):
            l1 = tf.keras.layers.Dense(256, activation='relu', name='term{}-1'.format(i))(layers[-1])
            l2 = tf.keras.layers.Dense(1, name='term{}-2'.format(i))(l1)
            logits.append(l2)
            output_layers[i] = l2
    else:
        dependencies = gogb.build_graph(term_indices_file)
        fc_layers = {}
        # get all top terms (without parents)
        terms_without_any_more_parents = [term for term in dependencies if not any(1 for parent in dependencies if parent in dependencies[term])]
        # as long as we have more terms without any more parents, loop
        ctr = 0
        while terms_without_any_more_parents:
            ctr+=1
            # create fully-connected layer using the layers[-1] and FC of previous parents
            this_term = terms_without_any_more_parents.pop(0)                                       # get a new term to add
            parents = dependencies[this_term]                                                       # get the parents of this term
            children = list({key for key in dependencies if this_term in dependencies[key]}) # get the children of this term
            if parents:
                prev_l = tf.keras.layers.Concatenate()([layers[-1]]+[fc_layers[parent] for parent in parents])
            else:
                prev_l = layers[-1]
            l1 = tf.keras.layers.Dense(256, activation='relu', name='term{}-1'.format(this_term))(prev_l)
            fc_layers[this_term] = l1                                                               # add this FC layer to fc_layers
            l2 = tf.keras.layers.Dense(1, name='term{}-2'.format(this_term))(l1)                    # create the logit neuron and add to logits and output_layers
            logits.append(l2)
            output_layers[this_term] = l2

            set_of_added_terms = set(terms_without_any_more_parents + list(fc_layers.keys()))       # create a set of all terms that have a FC already
            # check for each child if it is eligible --- i.e. if all of its parents have been covered already
            terms_without_any_more_parents.extend([child for child in children if
                                                                    all(term in set_of_added_terms for term in dependencies[child]) and
                                                                    child not in set_of_added_terms
                                                                    ])

        for term in range(n_of_outputs):
            children_terms = list({key for key in dependencies if term in dependencies[key]})
            if len(children_terms) == 0:
                output_layers[term] = logits[term]
            else:
                all_chldrn_l = tf.keras.layers.Concatenate()([logits[x] for x in [term]+children_terms])
                mx_l = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=1, keepdims=True), name='term{}-3'.format(term))(all_chldrn_l)
                output_layers[term] = mx_l

    printNeuralNet(layers)
    print('And then some output layers... ({})\n'.format(len(output_layers)))

    # The output layer here is returned as logits. Sigmoids are added in the TrainingProcedure.py file
    cc = tf.keras.layers.Concatenate(name='my_logits')(output_layers)
    print('{:35s} -> {}'.format(cc.name, cc.shape))

    # Create model
    model = tf.keras.Model(inputs=[X_input, seqlen_input, vec_input], outputs=cc, name='deepgo_model')
    return NetworkObject(model, X_input, seqlen_input, vec_input)


# Returns one of our models, according to the parameters. Explanation of the parameters can be found on top of this file
def buildMyNetwork(type, ngramsize, n_of_outputs, term_indices_file, filterSizes, filterAmounts, maxPoolSizes, sizeOfFCLayers,
                   dynMaxPoolSize, ppi_vectors, hierarchy, embeddingDepth, embeddingType, GRU_state_size, maxLength):

    # Define inputs
    X_input = tf.keras.Input(shape=(maxLength,), dtype=tf.int32, name='X_placeholder')
    seqlen_input = tf.keras.Input(shape=(), dtype=tf.int32, name='seqlen_placeholder')
    vec_input = tf.keras.Input(shape=(256,), dtype=tf.float32, name='vec_placeholder')

    layers = []
    vocab_size = 20**ngramsize + 1

    ### Embedding layer ###
    if embeddingType == 'onehot':
        # Create one-hot embedding matrix
        onehot_matrix = np.zeros((vocab_size, 20**ngramsize), dtype=np.float32)
        for i in range(20**ngramsize):
            onehot_matrix[i+1][i] = 1
        embedding_init = tf.keras.initializers.Constant(onehot_matrix)
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=20**ngramsize,
            embeddings_initializer=embedding_init,
            trainable=False,
            mask_zero=False,
            name='embedding_out'
        )
    elif embeddingType == 'trainable':
        embedding_init = tf.keras.initializers.RandomUniform(-0.05, 0.05)
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embeddingDepth,
            embeddings_initializer=embedding_init,
            trainable=True,
            mask_zero=True,
            name='embedding_out'
        )
    else:
        raise AssertionError(f'embeddingType {embeddingType} unknown')

    l = embedding_layer(X_input)
    layers.append(l)

    # Track sequence lengths through pooling
    seqlens = seqlen_input

    ### Convolutional, dropout and maxpool layers ###
    for idx, (f_size, f_amount, p_size) in enumerate(zip(filterSizes, filterAmounts, maxPoolSizes)):
        layers.append(tf.keras.layers.Conv1D(f_amount, f_size, padding='same', activation='relu')(layers[-1]))
        layers.append(tf.keras.layers.Dropout(0.2)(layers[-1]))  # Dropout rate will be controlled by training flag
        layers.append(tf.keras.layers.MaxPooling1D(p_size, p_size)(layers[-1]))
        seqlens = seqlens // p_size
    print(seqlens)

    ### Varying input strategy, depending on the type ###
    if type == 'D':
        # Dynamic max pooling - need to use Lambda layer for custom operation
        layers.append(tf.keras.layers.Lambda(
            lambda x: dynamic_max_pooling_with_overlapping_windows(x[0], x[1], fixed_output_size=dynMaxPoolSize),
            name='dynamic_max_pool'
        )([layers[-1], seqlens]))
    elif type == 'K':
        layers.append(tf.keras.layers.Permute((2, 1))(layers[-1]))
        layers.append(tf.keras.layers.Lambda(
            lambda x: tf.nn.top_k(x, k=dynMaxPoolSize, sorted=False)[0],
            name='k_max_pool'
        )(layers[-1]))
    elif type == 'O':
        pass  # do nothing special
    elif type == 'R':
        layers.append(tf.keras.layers.Lambda(
            lambda x: BidirectionalGRULayer(x[0], x[1], GRU_state_size),
            name='bidirectional_gru'
        )([layers[-1], seqlens]))
    elif type == 'M':
        pool_size = int(layers[-1].shape[1])
        layers.append(tf.keras.layers.MaxPooling1D(pool_size, pool_size)(layers[-1]))
    elif type == 'C':
        # Combined D+K
        D_layer = tf.keras.layers.Lambda(
            lambda x: dynamic_max_pooling_with_overlapping_windows(x[0], x[1], fixed_output_size=dynMaxPoolSize),
            name='dynamic_max_pool_c'
        )([layers[-1], seqlens])
        K_layer_pre = tf.keras.layers.Permute((2, 1))(layers[-1])
        K_layer = tf.keras.layers.Lambda(
            lambda x: tf.nn.top_k(x, k=dynMaxPoolSize, sorted=False)[0],
            name='k_max_pool_c'
        )(K_layer_pre)
        K_layer_T = tf.keras.layers.Permute((2, 1))(K_layer)
        layers.append(D_layer)
        layers.append(K_layer_T)
        layers.append(tf.keras.layers.Concatenate(axis=1)([D_layer, K_layer_T]))

    layers.append(tf.keras.layers.Flatten()(layers[-1]))

    ### Concatenate domain vectors if specified ###
    if ppi_vectors:
        layers.append(tf.keras.layers.Concatenate()([layers[-1], vec_input]))

    ### Build output layers ###
    logits = []
    output_layers = [None] * n_of_outputs
    if not hierarchy:
        for i in range(n_of_outputs):
            l1 = tf.keras.layers.Dense(sizeOfFCLayers, activation='relu', name='term{}-1'.format(i))(layers[-1])
            l2 = tf.keras.layers.Dense(1, name='term{}-2'.format(i))(l1)
            logits.append(l2)
            output_layers[i] = l2
    else:
        dependencies = gogb.build_graph(term_indices_file)
        fc_layers = {}
        # get all top terms (without parents)
        terms_without_any_more_parents = [term for term in dependencies if not any(1 for parent in dependencies if parent in dependencies[term])]
        # as long as we have more terms without any more parents, loop
        ctr = 0
        while terms_without_any_more_parents:
            ctr+=1
            # create fully-connected layer using the layers[-1] and FC of previous parents
            this_term = terms_without_any_more_parents.pop(0)                                       # get a new term to add
            parents = dependencies[this_term]                                                       # get the parents of this term
            children = list({key for key in dependencies if this_term in dependencies[key]})        # get the children of this term
            if parents:
                prev_l = tf.keras.layers.Concatenate()([layers[-1]]+[fc_layers[parent] for parent in parents])
            else:
                prev_l = layers[-1]
            l1 = tf.keras.layers.Dense(sizeOfFCLayers, activation='relu', name='term{}-1'.format(this_term))(prev_l)
            fc_layers[this_term] = l1                                                               # add this FC layer to fc_layers
            l2 = tf.keras.layers.Dense(1, name='term{}-2'.format(this_term))(l1)                    # create the logit neuron and add to logits and output_layers
            logits.append(l2)
            output_layers[this_term] = l2

            set_of_added_terms = set(terms_without_any_more_parents + list(fc_layers.keys()))       # create a set of all terms that have a FC already
            # check for each child if it is eligible --- i.e. if all of its parents have been covered already
            terms_without_any_more_parents.extend([child for child in children if
                                                                    all(term in set_of_added_terms for term in dependencies[child]) and
                                                                    child not in set_of_added_terms
                                                                    ])

        for term in range(n_of_outputs):
            children_terms = list({key for key in dependencies if term in dependencies[key]})
            if len(children_terms) == 0:
                output_layers[term] = logits[term]
            else:
                all_chldrn_l = tf.keras.layers.Concatenate()([logits[x] for x in [term]+children_terms])
                mx_l = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=1, keepdims=True), name='term{}-3'.format(term))(all_chldrn_l)
                output_layers[term] = mx_l

    printNeuralNet(layers)
    print('And then some output layers... ({})\n'.format(len(output_layers)))

    # The output layer here is returned as logits. Sigmoids are added in the TrainingProcedure.py file
    cc = tf.keras.layers.Concatenate(name='my_logits')(output_layers)
    print('{:35s} -> {}'.format(cc.name, cc.shape))

    # Create model
    model = tf.keras.Model(inputs=[X_input, seqlen_input, vec_input], outputs=cc, name='my_network_model')
    return NetworkObject(model, X_input, seqlen_input, vec_input)


def buildPPIOnlyNetwork(n_of_outputs, term_indices_file, sizeOfFCLayers, hierarchy, maxLength):
    # Define inputs
    X_input = tf.keras.Input(shape=(maxLength,), dtype=tf.int32, name='X_placeholder')
    seqlen_input = tf.keras.Input(shape=(), dtype=tf.int32, name='seqlen_placeholder')
    vec_input = tf.keras.Input(shape=(256,), dtype=tf.float32, name='vec_placeholder')

    layers = []
    layers.append(tf.keras.layers.Dense(sizeOfFCLayers, activation='relu')(vec_input))

    logits = []
    output_layers = [None] * n_of_outputs

    if not hierarchy:
        for i in range(n_of_outputs):
            l1 = tf.keras.layers.Dense(64, activation='relu', name='term{}-1'.format(i))(layers[-1])
            l2 = tf.keras.layers.Dense(1, name='term{}-2'.format(i))(l1)
            logits.append(l2)
            output_layers[i] = l2
    else:
        dependencies = gogb.build_graph(term_indices_file)
        fc_layers = {}
        # get all top terms (without parents)
        terms_without_any_more_parents = [term for term in dependencies if not any(1 for parent in dependencies if parent in dependencies[term])]
        # as long as we have more terms without any more parents, loop
        ctr = 0
        while terms_without_any_more_parents:
            ctr+=1
            # create fully-connected layer using the layers[-1] and FC of previous parents
            this_term = terms_without_any_more_parents.pop(0)                                       # get a new term to add
            parents = dependencies[this_term]                                                       # get the parents of this term
            children = list({key for key in dependencies if this_term in dependencies[key]}) # get the children of this term
            if parents:
                prev_l = tf.keras.layers.Concatenate()([layers[-1]]+[fc_layers[parent] for parent in parents])
            else:
                prev_l = layers[-1]
            l1 = tf.keras.layers.Dense(64, activation='relu', name='term{}-1'.format(this_term))(prev_l)
            fc_layers[this_term] = l1                                                               # add this FC layer to fc_layers
            l2 = tf.keras.layers.Dense(1, name='term{}-2'.format(this_term))(l1)                    # create the logit neuron and add to logits and output_layers
            logits.append(l2)
            output_layers[this_term] = l2

            set_of_added_terms = set(terms_without_any_more_parents + list(fc_layers.keys()))       # create a set of all terms that have a FC already
            # check for each child if it is eligible --- i.e. if all of its parents have been covered already
            terms_without_any_more_parents.extend([child for child in children if
                                                                    all(term in set_of_added_terms for term in dependencies[child]) and
                                                                    child not in set_of_added_terms
                                                                    ])

        for term in range(n_of_outputs):
            children_terms = list({key for key in dependencies if term in dependencies[key]})
            if len(children_terms) == 0:
                output_layers[term] = logits[term]
            else:
                all_chldrn_l = tf.keras.layers.Concatenate()([logits[x] for x in [term]+children_terms])
                mx_l = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=1, keepdims=True), name='term{}-4'.format(term))(all_chldrn_l)
                output_layers[term] = mx_l

    printNeuralNet(layers)
    print('And then some output layers... ({})\n'.format(len(output_layers)))

    # The output layer here is returned as logits. Sigmoids are added in the TrainingProcedure.py file
    cc = tf.keras.layers.Concatenate(name='my_logits')(output_layers)
    print('{:35s} -> {}'.format(cc.name, cc.shape))

    # Create model
    model = tf.keras.Model(inputs=[X_input, seqlen_input, vec_input], outputs=cc, name='ppi_only_model')
    return NetworkObject(model, X_input, seqlen_input, vec_input)


# Objects of this class hold a Keras model, as well as the input layers used in that model
class NetworkObject:
    def __init__(self, model, X_input, seqlen_input, vec_input):
        self.model = model
        self.X_input = X_input
        self.seqlen_input = seqlen_input
        self.vec_input = vec_input

    def getModel(self):
        return self.model

    def getNetwork(self):
        """Legacy method - returns a callable that mimics the old function-based network"""
        return lambda X, seqlens, vec: self.model([X, seqlens, vec])

    def getSeqLenPlaceholder(self):
        return self.seqlen_input

    def get_X_placeholder(self):
        return self.X_input

    def get_vec_placeholder(self):
        return self.vec_input

    def __call__(self, inputs, training=False):
        return self.model(inputs, training=training)
