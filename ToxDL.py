__author__ = 'jasper.zuallaert, Xiaoyong.Pan'

from DatasetCreator import createDatasets
from TestLauncher import runTest
from IntegratedGradientsRunner import runFromModel
from PosSeqFromSaliencyMapFile import selectPosSeqFromFile
from InterProVisualizer import runInterProVisualizer
from SequenceShowerAA import visualizeSaliencyMapFile
import InputManager as im
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np


### the created dataset is picked
NETWORK_SETTINGS_FILE = 'TestFiles/000_test.test'


def run_prediciton(testFile, predictions_save_dest = 'dl.score', save = True):
    test_dataset = im.getSequences_without_shuffle(testFile, 1, 1002, silent=True)

    # Load the saved Keras model
    model_path = 'parameters/test_200114-153051/test_200114-153051.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        # Try loading weights if full model not found
        # Would need to rebuild model architecture first
        raise FileNotFoundError(f"Model not found at {model_path}. Please ensure the model is saved in Keras format.")

    test_label, test_pred = [], []
    if save:
        a = open('dl.score', 'w')
    batches_done = False
    while not batches_done:
        batch_x, lengths_x, batch_y, vector_x, epoch_finished = test_dataset.next_batch_without_shuffle(512)

        # Convert to tensors
        batch_x_t = tf.constant(batch_x, dtype=tf.int32)
        vector_x_t = tf.constant(vector_x, dtype=tf.float32)
        lengths_x_t = tf.constant(lengths_x, dtype=tf.int32)

        # Forward pass
        logits = model([batch_x_t, lengths_x_t, vector_x_t], training=False)
        sigmoids = tf.nn.sigmoid(logits).numpy()

        for p, c in zip(sigmoids, batch_y):
            if save:
                print(','.join([str(x) for x in p]), file=a)
                print(','.join([str(x) for x in c]), file=a)
            test_label.append(c[0])
            test_pred.append(p[0])
        if epoch_finished:
            batches_done = True

    if save:
        a.close()
    return np.array(test_label), np.array(test_pred)


def run_motif_scan(testFile, saliencyMapFile):
    testset = im.getSequences(testFile, 1, 1002, silent=True)
    trainset = im.getSequences('/home/zzegs/workspace/dagw/toxicity_DL/data/train_data_file.dat.domain.toxin', 1, 1002, silent=True)

    # Load the saved Keras model
    model_path = '/home/zzegs/workspace/dagw/toxicity_DL/rr/BASF_code/parameters/seq_model/test_190705-110204.keras'
    model = tf.keras.models.load_model(model_path)

    runFromModel(0, model, trainset, testset, useRef=True, outF=saliencyMapFile)

    fastaFile, posSaliencyFile = selectPosSeqFromFile(saliencyMapFile)
    visualizeSaliencyMapFile(posSaliencyFile, 'seq_temp')


def run():
    # the training, validation and test set
    datafiles_tuple = ('datasets/train.fa.domain', 'datasets/valid.fa.domain', 'datasets/test.fa.domain', 'toxicity.indices')
    ### train network
    print('>>> TRAINING NETWORK...')
    results = []
    for i in range(10):
        model, trainset, testset, auROC, auPRC, F1score, MCC = runTest(NETWORK_SETTINGS_FILE, datafiles_tuple)
        # Clear session between runs to free memory
        tf.keras.backend.clear_session()
        results.append([auROC, auPRC, F1score, MCC])
    print('auROC, ', 'auPRC, ', 'F1socre, ', 'MCC')
    print(results)
    print('Mean results, auROC, auPRC, F1socre, MCC, of 10 runnning')
    print(np.mean(results, axis=0))
    ### build saliency map


if __name__ == "__main__":
    run()
