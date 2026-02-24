__author__ = 'jasper.zuallaert'
import os
# hide tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from InputManager import Dataset
import sys
import time
import Evaluation as eval

# Configure GPU memory growth at module load
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")

# Prepares a TrainingProcedure object for training, using a given network_object and a given training set, following
# the given parameters:
# - network_object: a NetworkObject object as returned by the functions in NetworkTopologyConstructor.py
# - train_dataset: an InputManager.Dataset object
# - valid_dataset: an InputManager.Dataset object
# - test_dataset: an InputManager.Dataset object
# - batch_size: integer
# - start_learning_rate: float
# - validationFunction: the metric which will be looked at during validation, to select the optimal model during
#                       training. Should be one of 'loss', 'f1'
# - update: a string indicating the update strategy. Should be one of 'momentum', 'rmsprop', 'adam'
# - dropoutRate: float
# - l1reg: l1reg multiplier, indicating whether or not L1 regularization should be applied on, and what the multiplier
#               should be if > 0:
#               a) the trainable embedding layer if it is specified
#               b) the first convolutional layer if no trainable embedding is used
# - lossFunction: the loss function type; should be one of 'default' (categorical crossentropy), 'weighted', 'focal'
#                 In the case of weighted loss, the inverse class frequency is used as a multiplier (see code below,
#                 still under experimentation)
class TrainingProcedure:
    def __init__(self, network_object, train_dataset, valid_dataset, test_dataset, batch_size, start_learning_rate,
                 validationFunction, update, dropoutRate, l1reg, lossFunction):
        self.validationFunction = validationFunction
        self.model = network_object.getModel()
        self.network_object = network_object
        self.n_of_output_classes = test_dataset.getClassCounts()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.dropoutRate = dropoutRate
        self.l1reg = l1reg
        self.lossFunction = lossFunction

        # Setup loss function
        if lossFunction == 'default':
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        elif lossFunction == 'weighted':
            class_counts = self.train_dataset.getCountsPerTerm()
            class_counts = np.maximum(class_counts, np.percentile(class_counts, 5))
            class_counts = np.max(class_counts) / class_counts
            self.class_weights = (class_counts / np.max(class_counts)).astype(np.float32)
            self.loss_fn = None  # Will compute manually
        elif lossFunction == 'focal':
            from Layers import focal_loss
            self.loss_fn = focal_loss
        else:
            raise ValueError(f"Unknown loss function: {lossFunction}")

        # Setup optimizer
        if update == 'momentum':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=start_learning_rate, momentum=0.9)
        elif update == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=start_learning_rate)
        elif update == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=start_learning_rate)
        else:
            raise Exception('Unknown update strategy declaration: {}'.format(update))

        self.total_parameters = self._print_num_params()

    # Prints the total number of trainable parameters
    def _print_num_params(self):
        total_parameters = 0
        for variable in self.model.trainable_variables:
            local_parameters = np.prod(variable.shape)
            total_parameters += local_parameters
        print('This network has {} trainable parameters.'.format(total_parameters))
        if total_parameters < 5000000 and sys.argv[0] != 'SingleTermWorkflow.py':
            print('total_parameters < 5000000 => model will be saved')
        return total_parameters

    def _compute_loss(self, y_true, logits):
        """Compute loss based on the configured loss function"""
        if self.lossFunction == 'default':
            return self.loss_fn(y_true, logits)
        elif self.lossFunction == 'weighted':
            # Weighted sigmoid cross entropy
            loss = tf.math.maximum(logits, 0) - logits * y_true + tf.math.log(1 + tf.math.exp(-tf.math.abs(logits)))
            weighted_loss = self.class_weights * loss
            return tf.reduce_mean(weighted_loss)
        elif self.lossFunction == 'focal':
            return self.loss_fn(logits, y_true)

    @tf.function
    def _train_step(self, batch_x, batch_y, batch_vector, batch_lengths):
        """Single training step with gradient tape"""
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.model([batch_x, batch_lengths, batch_vector], training=True)
            loss = self._compute_loss(batch_y, logits)

            # Add L1 regularization if specified
            if self.l1reg > 0:
                # Apply L1 reg to first trainable layer
                if len(self.model.trainable_variables) > 0:
                    loss = loss + self.l1reg * tf.reduce_sum(tf.abs(self.model.trainable_variables[0]))

        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    # This function trains the network specified initially, with the datasets specified initially
    # Parameters:
    # - epochs: the number of epochs that should be trained
    # Note: based on the initially specified validationFunction, the best model will be used for the final predictions
    def trainNetwork(self, epochs):
        predictions_save_dest = 'predictions/test_{}.txt'.format(time.strftime('%y%m%d-%H%M%S'))
        parameters_save_dest = 'parameters/test_{}'.format(time.strftime('%y%m%d-%H%M%S'))

        self._printOutputClasses(self.train_dataset, 'Training')
        self._printOutputClasses(self.valid_dataset, 'Valid')
        self._printOutputClasses(self.test_dataset, 'Test')

        print(' {:^5} | {:^14} | {:^14} | {:^14} | {:^14} | {:^14} | {:^14} | {:^14} | {:^12} | {:^12}'.format('epoch','train loss','valid loss','tr Fmax','va Fmax','te Fmax','te avgPr','te avgSn','total time','train time'))
        print('-{:-^6}+{:-^16}+{:-^16}+{:-^16}+{:-^16}+{:-^16}+{:-^16}+{:-^16}+{:-^12}-{:-^13}-'.format('','','','','','','','','','','',''))

        ### Pre training, output ##
        best_valid_score = 999999 if self.validationFunction == 'loss' or self.validationFunction == 'fpr' else -1

        t1 = time.time()
        tr_loss, tr_Fmax, tr_avgPr, tr_avgSn = self._evaluateSet(-1, self.train_dataset, 512)
        va_loss, va_Fmax, va_avgPr, va_avgSn = self._evaluateSet(-1, self.valid_dataset, 512)
        te_loss, te_Fmax, te_avgPr, te_avgSn = -1, -1, -1, -1

        print(' {:5d} |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {:4.2f}s     |   {:4.2f}s   '.format(0,tr_loss,va_loss,tr_Fmax,va_Fmax,te_Fmax,te_avgPr,te_avgSn,time.time()-t1,0))

        ### train for each epoch ###
        for epoch in range(1, epochs):
            sys.stdout.flush()
            epoch_start_time = time.time()

            epoch_finished = False
            trainstart = time.time()
            ### train for each batch in this epoch ###
            while not epoch_finished:
                batch_x, lengths_x, batch_y, vector_x, epoch_finished = self.train_dataset.next_batch(self.batch_size)

                # Convert to tensors
                batch_x = tf.constant(batch_x, dtype=tf.int32)
                batch_y = tf.constant(batch_y, dtype=tf.float32)
                vector_x = tf.constant(vector_x, dtype=tf.float32)
                lengths_x = tf.constant(lengths_x, dtype=tf.int32)

                self._train_step(batch_x, batch_y, vector_x, lengths_x)

            trainstop = time.time()

            ### !!! for time-saving purposes, I only calculate the validation metrics - the rest is filled in with -1 ###
            tr_loss, tr_Fmax, tr_avgPr, tr_avgSn = -1,-1,-1,-1
            va_loss, va_Fmax, va_avgPr, va_avgSn = self._evaluateSet(epoch, self.valid_dataset, 1024)
            te_loss, te_Fmax, te_avgPr, te_avgSn = -1,-1,-1,-1

            print_message = ''
            valid_metric_score = va_loss if self.validationFunction == 'loss' else va_Fmax if self.validationFunction == 'f1' else None
            ### if new best validation result - store the parameters + generate predictions on test set ###
            if valid_metric_score != None and self._compareValidMetrics(valid_metric_score, best_valid_score):
                best_valid_score = valid_metric_score
                self._storeNetworkParameters(parameters_save_dest)
                self._writePredictions(predictions_save_dest)
                print_message = '-> New best valid.'

            print(' {:5d} |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {:4.2f}s     |   {:4.2f}s   {}'.format(epoch,tr_loss,va_loss,tr_Fmax,va_Fmax,te_Fmax,te_avgPr,te_avgSn,time.time()-epoch_start_time,trainstop-trainstart,print_message))

        print("Finished")
        print('Parameters should\'ve been stored in {}'.format(parameters_save_dest))

        ### Generate predictions to show at the end of the file, using Evaluation.py  ###
        auROC, auPRC, Fmax, mcc = eval.run_eval_per_term(predictions_save_dest)
        if self.n_of_output_classes > 1:
            eval.run_eval_per_protein(predictions_save_dest)
        return self.model, auROC, auPRC, Fmax, mcc

    # Generate the losses, f1 scores and other metrics for a given dataset
    def _evaluateSet(self, epoch, dataset: Dataset, batch_size, threshold_range = 20):
        losses = []
        F_per_thr = []
        avgPr_per_thr = []
        avgSn_per_thr = []

        ### go over each batch and store the losses ###
        batches_done = False
        while not batches_done:
            batch_x, lengths_x, batch_y, vector_x, epoch_finished = dataset.next_batch(batch_size)

            # Convert to tensors
            batch_x_t = tf.constant(batch_x, dtype=tf.int32)
            batch_y_t = tf.constant(batch_y, dtype=tf.float32)
            vector_x_t = tf.constant(vector_x, dtype=tf.float32)
            lengths_x_t = tf.constant(lengths_x, dtype=tf.int32)

            # Forward pass
            logits = self.model([batch_x_t, lengths_x_t, vector_x_t], training=False)
            loss_batch = self._compute_loss(batch_y_t, logits).numpy()
            losses.extend([loss_batch] * len(batch_x))
            if epoch_finished:
                batches_done = True

        ### at the desired epochs (currently: all), do the calculations ###
        if epoch >= 0 and epoch % 1 == 0:
            ### for every threshold, calculate pr, sn, fscore ###
            for t in range(threshold_range):
                threshold = t/threshold_range
                prSum = 0.0
                snSum = 0.0
                n_of_samples_predicted_pos = 0
                batches_done = False
                while not batches_done:
                    batch_x, lengths_x, batch_y, vector_x, epoch_finished = dataset.next_batch(batch_size)

                    # Convert to tensors
                    batch_x_t = tf.constant(batch_x, dtype=tf.int32)
                    batch_y_t = tf.constant(batch_y, dtype=tf.float32)
                    vector_x_t = tf.constant(vector_x, dtype=tf.float32)
                    lengths_x_t = tf.constant(lengths_x, dtype=tf.int32)

                    # Forward pass and get sigmoid
                    logits = self.model([batch_x_t, lengths_x_t, vector_x_t], training=False)
                    sigmoid_out = tf.nn.sigmoid(logits)
                    preds = tf.math.ceil(sigmoid_out - threshold)

                    tp_res = tf.reduce_sum((batch_y_t + preds) // 2, axis=1).numpy()
                    n_of_pos_res = tf.reduce_sum(batch_y_t, axis=1).numpy()
                    predicted_pos_res = tf.reduce_sum(preds, axis=1).numpy()

                    for tp, n_pos, pred_pos in zip(tp_res, n_of_pos_res, predicted_pos_res):
                        if tp:
                            n_of_samples_predicted_pos += 1
                            prSum += tp / pred_pos
                            snSum += tp / n_pos

                    if epoch_finished:
                        batches_done = True

                avgPr = prSum / max(1, n_of_samples_predicted_pos)
                avgSn = snSum / len(dataset)
                avgPr_per_thr.append(avgPr)
                avgSn_per_thr.append(avgSn)
                F_per_thr.append(2*avgPr*avgSn/(avgPr+avgSn) if avgPr+avgSn > 0 else 0.0)
            Fmax_index = np.argmax(F_per_thr)
            return np.average(losses), F_per_thr[Fmax_index], avgPr_per_thr[Fmax_index], avgSn_per_thr[Fmax_index]
        else:
            return np.average(losses), -1, -1, -1

    # Store network parameters using Keras model save
    def _storeNetworkParameters(self, saveToDir):
        if self.total_parameters < 5000000:
            try:
                if not os.path.exists(saveToDir):
                    os.makedirs(saveToDir)
                # Save in Keras format
                self.model.save(saveToDir + '/' + saveToDir[saveToDir.rfind('/')+1:] + '.keras')
                # Also save weights separately for compatibility
                self.model.save_weights(saveToDir + '/' + saveToDir[saveToDir.rfind('/')+1:] + '_weights.h5')
            except Exception as e:
                print('SOMETHING WENT WRONG WITH STORING PARAMETERS!!')
                print(e)
                print(sys.exc_info())

    # Writes predictions to a file, to be evaluated by Evaluation.py afterwards
    def _writePredictions(self, predictions_save_dest):
        # Ensure predictions directory exists
        os.makedirs(os.path.dirname(predictions_save_dest), exist_ok=True)
        a = open(predictions_save_dest, 'w')
        batches_done = False
        while not batches_done:
            batch_x, lengths_x, batch_y, vector_x, names, epoch_finished = self.test_dataset.next_batch_without_shuffle(512)

            # Convert to tensors
            batch_x_t = tf.constant(batch_x, dtype=tf.int32)
            vector_x_t = tf.constant(vector_x, dtype=tf.float32)
            lengths_x_t = tf.constant(lengths_x, dtype=tf.int32)

            # Forward pass and get sigmoid
            logits = self.model([batch_x_t, lengths_x_t, vector_x_t], training=False)
            sigmoids = tf.nn.sigmoid(logits).numpy()

            for p, c, n in zip(sigmoids, batch_y, names):
                print(','.join([str(x) for x in p]), file=a)
                print(','.join([str(x) for x in c]), file=a)
                print(n, file=a)
            if epoch_finished:
                batches_done = True
        a.close()

    # Prints the information about the dataset in input
    def _printOutputClasses(self, dataset, label):
        print(f'{label} set:')
        counts = dataset.getClassCounts()
        if counts == 1:
            print(f'Number of positives: {dataset.getPositiveCount()}')
            print(f'Number of negatives: {dataset.getNegativeCount()}')
        else:
            print('Number of {} classes: {}'.format(label, counts))
            print('Number of {} samples: {}'.format(label, len(dataset)))

    # Compares two validation metrics
    def _compareValidMetrics(self, new, old):
        if self.validationFunction == 'loss':
            return new < old
        else:
            return new > old
