import time

import numpy as np
import torch as t

from q1_softmax import softmax
from q1_softmax import cross_entropy_loss
from model import Model
from utils.general_utils import get_minibatches
import torch.optim as opt

class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_samples = 1024
    n_features = 100
    n_classes = 5
    batch_size = 64
    n_epochs = 50
    lr = 1e-4


class SoftmaxModel(Model):
    """Implements a Softmax classifier with cross-entropy loss."""

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.

        These placeholders are used as inputs by the rest of the model building
        and will be fed data during training.

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of shape
                                              (batch_size, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape
                                              (batch_size, n_classes), type tf.int32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
        """
        ### YOUR CODE HERE
        # self.inputs = t.zeros(self.config.batch_size, self.config.n_features, dtype=t.float32)
        # self.labels = t.zeros(self.config.batch_size, self.config.n_classes, dtype=t.int32)
        self.inputs = None
        self.labels = None
        self.W = t.rand((self.config.n_features, self.config.n_classes), requires_grad=True)
        self.b = t.rand((self.config.n_classes, ), requires_grad=True)
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If label_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be the placeholder
                tensors created in add_placeholders.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE
        ### END YOUR CODE
        return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. In this case, the transformation is a linear layer plus a
        softmax transformation:

        yhat = softmax(xW + b)

        Hint: The input x will be passed in through self.input_placeholder. Each ROW of
              self.input_placeholder is a single example. This is usually best-practice for
              tensorflow code.
        Hint: Make sure to create tf.Variables as needed.
        Hint: For this simple use-case, it's sufficient to initialize both weights W
                    and biases b with zeros.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        ### YOUR CODE HERE
        # W = t.rand((self.config.n_features, self.config.n_classes), requires_grad=True)
        # b = t.rand((self.config.n_classes, ), requires_grad=True)
        z = t.mm(self.inputs, self.W) + self.b
        pred = softmax(z)
        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.

        Hint: Use the cross_entropy_loss function we defined. This should be a very
                    short function.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE
        loss = cross_entropy_loss(self.labels, pred)
        ### END YOUR CODE
        return loss

    def add_training_op(self):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/api_docs/python/tf/train/Optimizer

        for more information. Use the learning rate from self.config.

        Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
                    Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE
        train_op = opt.SGD([self.W, self.b], lr=self.config.lr)
        ### END YOUR CODE
        return train_op

    def run_epoch(self, inputs, labels):
        """Runs an epoch of training.

        Args:
            sess: tf.Session() object
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size):
            n_minibatches += 1
            total_loss += self.train_on_batch(t.tensor(input_batch, dtype=t.float32),
             t.tensor(labels_batch, dtype=t.int32))
        return total_loss / n_minibatches

    def fit(self, inputs, labels):
        """Fit model on provided data.

        Args:
            sess: tf.Session()
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            losses: list of loss per epoch
        """
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(inputs, labels)
            duration = time.time() - start_time
            print('Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration))
            losses.append(average_loss)
        return losses

    def __init__(self, config):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()


def test_softmax_model():
    """Train softmax model for a number of steps."""
    config = Config()

    # Generate random data to train the model on
    np.random.seed(1234)
    inputs = np.random.rand(config.n_samples, config.n_features)
    labels = np.zeros((config.n_samples, config.n_classes), dtype=np.int32)
    labels[:, 0] = 1

    # 开始构建模型
    model = SoftmaxModel(config)
    losses = model.fit(inputs, labels)
    # If ops are implemented correctly, the average loss should fall close to zero
    # rapidly.
    assert losses[-1] < .5
    print ("Basic (non-exhaustive) classifier tests pass")

if __name__ == "__main__":
    test_softmax_model()
