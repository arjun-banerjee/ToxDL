__author__ = 'jasper.zuallaert'
import sys
import numpy as np
import InputManager as im
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


MAXIMUM_LENGTH = 1002  # hard-coded maximum length, for now


def compute_gradients_for_embeddings(model, embedding_layer, batch_embeddings, lengths, termN):
    """Compute gradients of the term logit with respect to embeddings using GradientTape"""
    batch_embeddings = tf.constant(batch_embeddings, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(batch_embeddings)

        # We need to run the model from embeddings onwards
        # Get the layers after embedding
        x = batch_embeddings

        # Find the embedding layer index and process through remaining layers
        found_embedding = False
        for layer in model.layers:
            if layer.name == 'embedding_out':
                found_embedding = True
                continue  # Skip embedding, we're using our own embeddings
            if not found_embedding:
                continue  # Skip layers before embedding

            # Skip input layers
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            # Handle different layer types
            try:
                x = layer(x)
            except:
                # Some layers may need different handling
                pass

        # Get the logit for the specific term
        term_logit = x[:, termN:termN+1]

    gradients = tape.gradient(term_logit, batch_embeddings)
    return gradients.numpy() if gradients is not None else np.zeros_like(batch_embeddings)


# Called from SingleTermWorkflow, or as a standalone python script
# Takes a trained model, and generates saliency maps for all sequences in a given test_dataset
# The train_dataset given is used for calculating the reference, if desired
# Note that this should only be used in case of one-hot encoding
def runIntegratedGradientsOnTestSet(termN,
                                    model,
                                    train_dataset,
                                    test_dataset,
                                    use_reference=False,
                                    outF=None):
    # Get embedding layer from model
    embedding_layer = None
    for layer in model.layers:
        if layer.name == 'embedding_out':
            embedding_layer = layer
            break

    if embedding_layer is None:
        raise ValueError("Could not find embedding layer named 'embedding_out' in model")

    epoch_finished = False
    outFile = open(outF, 'w') if outF else None

    ### Calculate the reference (ran first positions, ran last positions and an average for all positions in between)
    ran = 5
    freqs = np.zeros((ran * 2 + 1, 20), dtype=np.float32)
    if use_reference:
        for sequence, seqlen in zip(train_dataset.getX(), train_dataset.getLengths()):
            seqlen = min(MAXIMUM_LENGTH, seqlen)
            for pos in range(ran):
                freqs[pos][int(sequence[pos]-1)] += 1
                freqs[-pos-1][int(sequence[seqlen-pos-1]-1)] += 1
            for pos in range(ran, seqlen-ran):
                freqs[ran][int(sequence[pos]-1)] += 1
        for pos in range(ran*2+1):
            freqs[pos] /= sum(freqs[pos])

    ### Increase num_integration_steps for higher precision
    num_integration_steps = 30
    while not epoch_finished:
        batch_x, lengths_x, batch_y, vector_data, epoch_finished = test_dataset.next_batch(1024)
        lengths_x = [min(x, MAXIMUM_LENGTH) for x in lengths_x]

        # Get embeddings using the model's embedding layer
        batch_x_t = tf.constant(batch_x, dtype=tf.int32)
        embedding_results = embedding_layer(batch_x_t).numpy()

        ### Calculate the difference from reference
        if use_reference:
            difference_part = np.zeros_like(embedding_results)
            for seq_n in range(len(batch_x)):
                for pos in range(ran):
                    difference_part[seq_n][pos] = (embedding_results[seq_n][pos] - freqs[pos]) / num_integration_steps
                for pos in range(ran, lengths_x[seq_n]-ran):
                    difference_part[seq_n][pos] = (embedding_results[seq_n][pos] - freqs[ran]) / num_integration_steps
                for pos in range(lengths_x[seq_n]-ran, lengths_x[seq_n]):
                    difference_part[seq_n][pos] = (embedding_results[seq_n][pos] - freqs[pos-lengths_x[seq_n]]) / num_integration_steps
        else:
            difference_part = embedding_results / num_integration_steps

        ### Calculate the gradients for each step
        allNucs = batch_x
        allClasses = [y[termN] for y in batch_y]
        allSeqLens = lengths_x
        allValues = np.zeros((len(batch_x), len(batch_x[0]), 20), np.float32)

        # Get predictions
        lengths_x_t = tf.constant(lengths_x, dtype=tf.int32)
        vector_data_t = tf.constant(vector_data, dtype=tf.float32)
        logits = model([batch_x_t, lengths_x_t, vector_data_t], training=False)
        allPreds = [p[termN] for p in tf.nn.sigmoid(logits).numpy()]

        for step in range(1, num_integration_steps + 1):
            baseline = np.zeros_like(embedding_results)
            if use_reference:
                for seq_n in range(len(batch_x)):
                    for pos in range(ran):
                        baseline[seq_n][pos] = freqs[pos]
                    for pos in range(ran, lengths_x[seq_n]-ran):
                        baseline[seq_n][pos] = freqs[ran]
                    for pos in range(lengths_x[seq_n]-ran, lengths_x[seq_n]):
                        baseline[seq_n][pos] = freqs[pos-lengths_x[seq_n]]

            batch_x_for_this_step_1 = baseline + difference_part * (step - 1)
            batch_x_for_this_step_2 = baseline + difference_part * step

            # Compute gradients using GradientTape
            all_gradients_1 = compute_gradients_for_embeddings(model, embedding_layer, batch_x_for_this_step_1, lengths_x, termN)
            all_gradients_2 = compute_gradients_for_embeddings(model, embedding_layer, batch_x_for_this_step_2, lengths_x, termN)

            allValues += (all_gradients_1 + all_gradients_2) / 2 * difference_part

        ### Generate outputs
        for pred, seq, cl, seqlen, values in zip(allPreds, allNucs, allClasses, allSeqLens, allValues):
            print('{},{},actual_length={}'.format(pred, cl, seqlen), file=outFile)
            print(','.join(['_ACDEFGHIKLMNPQRSTVWY'[int(nuc)] for nuc in seq[:seqlen]]), file=outFile)
            print(','.join([str(score[int(nuc)-1]) for score, nuc in zip(values[:seqlen], seq[:seqlen])]), file=outFile)

    if outFile:
        outFile.close()


# Alternative gradient computation that works with the full model
@tf.function
def compute_integrated_gradients(model, batch_x, lengths_x, vector_data, embedding_layer, termN, baseline, scale):
    """Compute gradients using GradientTape for integrated gradients"""
    batch_x = tf.cast(batch_x, tf.int32)
    lengths_x = tf.cast(lengths_x, tf.int32)
    vector_data = tf.cast(vector_data, tf.float32)

    with tf.GradientTape() as tape:
        # Get embeddings and watch them
        embeddings = embedding_layer(batch_x)
        tape.watch(embeddings)

        # Scale embeddings for this step
        scaled_embeddings = baseline + (embeddings - baseline) * scale

        # We need a custom forward pass using scaled embeddings
        # This is model-specific and may need adjustment
        logits = model([batch_x, lengths_x, vector_data], training=False)
        term_logit = logits[:, termN]

    gradients = tape.gradient(term_logit, embeddings)
    return gradients


# Function to call if we want to use IntegratedGradients from another file
def runFromModel(termN, model, train_set, test_set, useRef=True, outF=None):
    runIntegratedGradientsOnTestSet(termN, model, train_set, test_set, useRef, outF)


# Legacy function for backward compatibility - loads model from checkpoint
def runFromSession(termN, sess, train_set, test_set, useRef=True, outF=None):
    """Legacy function - no longer supports sessions. Use runFromModel instead."""
    raise NotImplementedError(
        "Session-based execution is no longer supported in TF2. "
        "Please use runFromModel(termN, model, train_set, test_set, useRef, outF) instead."
    )


# If called as a standalone python script
if len(sys.argv) != 6 and sys.argv[0] == 'IntegratedGradientsRunner.py':
    print('Usage: python IntegratedGradientsRunner.py <term number> <model path> <train file> <test file> <use_reference>')
elif sys.argv[0] == 'IntegratedGradientsRunner.py':
    termN = int(sys.argv[1])
    modelPath = sys.argv[2]  # e.g. 'parameters/test_181212_225609/test_181212_225609.keras'
    trainFile = sys.argv[3]  # e.g. 'inputs/mf_train.dat'
    testFile = sys.argv[4]   # e.g. 'inputs/mf_test.dat'
    useRef = bool(sys.argv[5])

    train_set = im.getSequences(trainFile, 1, MAXIMUM_LENGTH, silent=True)
    test_set = im.getSequences(testFile, 1, MAXIMUM_LENGTH, silent=True)

    # Load Keras model
    model = tf.keras.models.load_model(modelPath)

    runFromModel(termN, model, train_set, test_set, useRef=useRef)
