#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    epsilon = 10e-7
    ### YOUR CODE HERE
    x = x / (np.sqrt(np.apply_along_axis(lambda x: x.dot(x.T), axis=1, arr=x))[:, None] + epsilon)
    ### END YOUR CODE

    return x

def test_normalize_rows():
    print ("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print (x)
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    # target是指公式中下标为o的那个，在skipgram
    v_hat = predicted
    Pred = softmax(np.dot(outputVectors, v_hat))  #注意到每行代表一个词向量，与文档中恰好相反
    
    cost = -np.log(Pred[target])
    
    Pred[target] -= 1.
    # 关于V的梯度
    gradPred = np.dot(outputVectors.T, Pred)
    # 关于U的梯度
    grad = np.outer(Pred, v_hat)
    
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices

# 注：自己这个函数写的太乱了，大家可以参考下我下面注释掉那个
# 也可以去看自己举荐的那个2.7版本的solution,两者是一样的
# 本来自己最开始想实现向量化运算，结果发现有重复的词，导致梯度只更改了一次
def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    
    #更改了一下原始定义，indices里面存放负样本的下标,并且负采样会采集重复样本
    indices = []
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    gradPred = np.zeros(predicted.shape)
    grad = np.zeros(outputVectors.shape)        #没有采样到的样本梯度为0
    grad1 = np.zeros(outputVectors.shape)
    #避免重复运算
    z1 = sigmoid(np.dot(outputVectors[target, :], predicted))
    z2 = sigmoid(np.dot(-outputVectors[indices, :], predicted))
    
    cost = np.sum(-np.log(sigmoid(np.dot(-outputVectors[indices, :], predicted)))) - np.log(z1)

    gradPred = (z1 - 1) * outputVectors[target, :] - np.dot(outputVectors[indices, :].T, z2 - 1)  #虽然有重复样本，但是不会对其产生影响                  
#     grad1[indices]  = -np.dot((z2 - 1)[:, None], predicted[None, :])          #存在重复样本，会将结果进行覆盖，只能进行for循环
    for k in indices:
        z = sigmoid(np.dot(outputVectors[k], predicted))
        grad[k] += predicted * z

    grad[target] = (z1 - 1) * predicted   
#     ### END YOUR CODE

    return cost, gradPred, grad

# def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
#                                K=10):
#     """ Negative sampling cost function for word2vec models
#     Implement the cost and gradients for one predicted word vector
#     and one target word vector as a building block for word2vec
#     models, using the negative sampling technique. K is the sample
#     size.
#     Note: See test_word2vec below for dataset's initialization.
#     Arguments/Return Specifications: same as softmaxCostAndGradient
#     """

#     # Sampling of indices is done for you. Do not modify this if you
#     # wish to match the autograder and receive points!
#     indices = [target]
#     indices.extend(getNegativeSamples(target, dataset, K))

#     ### YOUR CODE HERE
#     grad = np.zeros(outputVectors.shape)
#     gradPred = np.zeros(predicted.shape)
#     cost = 0
#     z = sigmoid(np.dot(outputVectors[target], predicted))

#     cost -= np.log(z)
#     grad[target] += predicted * (z - 1.0)
#     gradPred += outputVectors[target] * (z - 1.0)

#     for k in range(K):
#         samp = indices[k + 1]
#         z = sigmoid(np.dot(outputVectors[samp], predicted))
#         cost -= np.log(1.0 - z)
#         grad[samp] += predicted * z
#         gradPred += outputVectors[samp] * z
#     ### END YOUR CODE

#     return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    center = tokens[currentWord]
    predicted = inputVectors[center, :]
    
    for context in contextWords:
        target = tokens[context]
        once_cost, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
        cost += once_cost
        gradIn[center, :] += gradPred
        gradOut += grad
    
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    predicted_indices = [tokens[i] for i in contextWords]
    # 在这里并没有进行取平均的操作，文档中没有这个要求，但是课程中在讲解CBOW时有提及
    # 当然这里取平均之后也是可行的，亲测有效^_^
    predicted = np.sum(inputVectors[predicted_indices], axis=0)
    target = tokens[currentWord]
    cost, gradin, gradOut = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
    #注意下面是加，而不是赋值，因为同一个样本重复出现,山下文中可能出现相同的词汇
    for i in predicted_indices:
        gradIn[i] += gradin
        
    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:int(N/2),:]
    outputVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:int(N/2), :] += gin / batchsize / denom
        grad[int(N/2):, :] += gout / batchsize / denom

    return cost, grad
def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print ("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print ("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print ("\n=== Results ===")
    print (skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
            dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (skipgram("c", 1, ["a", "b"],
            dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
            negSamplingCostAndGradient))
    print (cbow("a", 2, ["a", "b", "c", "a"],
            dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (cbow("a", 2, ["a", "b", "a", "c"],
            dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
            negSamplingCostAndGradient))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
