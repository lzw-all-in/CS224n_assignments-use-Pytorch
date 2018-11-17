import pickle
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import torch as t
import torch.optim as opt
from model import Model
from q2_initialization import xavier_weight_init
import sys
sys.path.append('./utils/') # 从下一级导入
from parser_utils import minibatches, load_and_preprocess_data


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_features = 36
    n_classes = 3
    dropout = 0.5  # (p_drop in the handout)
    embedding_size = 50
    hidden_size = 200
    labmda = 10e-7  # Chen & Manning 论文中的参数值
    batch_size = 1024
    n_epochs = 10
    lr = 0.0005


class ParserModel(Model):
    
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict which transition should be applied to a given partial parse
    configuration.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        ### YOUR CODE HERE
        self.inputs = None
        self.labels = None
        self.dropout = None
        # freze参数代表在训练中进不进行更新,False代表要更新
        xavier = xavier_weight_init()
        self.embedded = nn.Embedding.from_pretrained(t.from_numpy(self.pretrained_embeddings), freeze=False)
        self.b1 = t.zeros((self.config.hidden_size, ), requires_grad=True)
        self.b2 = t.zeros((self.config.n_classes, ), requires_grad=True)
        self.W = xavier((self.config.n_features * self.config.embedding_size, self.config.hidden_size))
        self.U = xavier((self.config.hidden_size, self.config.n_classes))
        # 由于自己一开始忘记设置为True导致dev上UAS只有15
        self.W.requires_grad = True
        self.U.requires_grad = True
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }


        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE
        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates a tf.Variable and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/tf/reshape

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        ### YOUR CODE HERE
        embeddings = self.embedded(self.inputs).view(-1, self.config.n_features * self.config.embedding_size)
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self, is_train=True):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        Note that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        Use the initializer from q2_initialization.py to initialize W and U (you can initialize b1
        and b2 with zeros)

        Hint: Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
              Therefore the keep probability should be set to the value of
              (1 - self.dropout_placeholder)

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """

        x = self.add_embedding()
        ### YOUR CODE HERE
        h = F.relu(t.mm(x, self.W) + self.b1)
        h_drop = F.dropout(h, self.config.dropout, training=is_train)
        pred = t.mm(h_drop, self.U) + self.b2 
        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, self.labels)
        ### END YOUR CODE
        return loss

    def add_training_op(self):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/api_docs/python/tf/train/Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Use the learning rate from self.config.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE
        # Pytorch已经把L2正则化写入到了优化器里面，自己找了半天没有找到
        # 这里weight_decay就是l2的lambda
        train_op = opt.Adam([self.W, self.b1, self.U, self.b2,
                                 self.embedded.weight], lr=self.config.lr, weight_decay=self.config.labmda)
        ### END YOUR CODE
        return train_op

    def train_on_batch(self, inputs_batch, labels_batch):
        self.inputs = t.from_numpy(inputs_batch)
        # 由于Pytorch的交叉熵传入的target需要是一维的索引变量
        # 所以需要将one hot转变为索引
        self.labels = t.from_numpy(labels_batch).long().nonzero()[:, 1]
        self.train_op.zero_grad()
        loss = self.add_loss_op(self.add_prediction_op())
        loss.backward()
        self.train_op.step()

        return loss.item()

    def run_epoch(self, parser, train_examples, dev_set):
        n_minibatches = 1 + len(train_examples) / self.config.batch_size
        loss = 0.0
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(train_x,
                                    train_y)
            
        print("train loss : ", loss)
        print ("Evaluating on dev set")
        dev_UAS, _ = parser.parse(dev_set)
        print ("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
        return dev_UAS

    def fit(self, parser, save, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print( "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(parser, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if save:
                    t.save(self.W, './data/weights/W1.pth')
                    t.save(self.U, './data/weights/U1.pth')
                    t.save(self.embedded, './data/weights/embedded1.pth')
                    t.save(self.b1, './data/weights/b1_ori.pth')
                    t.save(self.b2, './data/weights/b2_ori.pth')
            print()

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()


def main(debug=True):
    print (80 * "=")
    print( "INITIALIZING")
    print (80 * "=")
    config = Config()
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    print ("Building model...",)
    start = time.time()
    model = ParserModel(config, embeddings)
    parser.model = model
    print ("took {:.2f} seconds\n".format(time.time() - start))
    print( 80 * "=")
    print( "TRAINING")
    print( 80 * "=")
    save = False if debug else True
    model.fit(parser, save, train_examples, dev_set)

    if not debug:
        print (80 * "=")
        print ("TESTING")
        print (80 * "=")
        print ("Restoring the best model weights found on the dev set")
        model.W = t.load('./data/weights/W1.pth')
        model.U = t.load('./data/weights/U1.pth')
        model.embedded = t.load('./data/weights/embedded1.pth')
        model.b1 = t.load('./data/weights/b1_ori.pth')
        model.b2 = t.load('./data/weights/b2_ori.pth')
        print ("Final evaluation on test set",)
        UAS, dependencies = parser.parse(test_set)
        print ("- test UAS: {:.2f}".format(UAS * 100.0))
        print ("Writing predictions")
        with open('q2_test.predicted.pkl', 'wb') as f:
            pickle.dump(dependencies, f, 1)
        print ("Done!")


if __name__ == '__main__':
    main(debug=False)

