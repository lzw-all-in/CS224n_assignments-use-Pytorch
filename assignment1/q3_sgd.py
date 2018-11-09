#!/usr/bin/env python

SAVE_PARAMS_EVERY = 5000

import glob
import random
import numpy as np
import os.path as op
import pickle


def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets
    iter_ation start.
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter_ = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter_ > st):
            st = iter_

    if st > 0:
        print(st)
        with open("saved_params_%d.npy" % st, "rb") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter_, params):
    with open("saved_params_%d.npy" % iter_, "wb") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a cost and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iter_ations -- total iter_ations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iter_ations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """

    # Anneal learning rate every several iter_ations
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter_, oldx, state = load_saved_params()
        if start_iter_ > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter_ / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter_ = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter_ in range(start_iter_ + 1, iterations + 1):
        # Don't forget to apply the postprocessing after every iteration!
        # You might want to print the progress every few iterations.

        cost = None
        ### YOUR CODE HERE
        cost, grad = f(x)
        x -= step * grad
        x = postprocessing(x)
        ### END YOUR CODE

        if iter_ % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
                
            print("iter_ %d: %f" % (iter_, expcost))
        try:
            if iter_ % SAVE_PARAMS_EVERY == 0 and useSaved:
                save_params(iter_, x)

            if iter_ % ANNEAL_EVERY == 0:
                step *= 0.5
        except Exception as e:
            print(str(e))
    return x


def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 1 result:", t1)
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("test 2 result:", t2)
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 3 result:", t3)
    assert abs(t3) <= 1e-6

    print("")

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q3_sgd.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE



if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
