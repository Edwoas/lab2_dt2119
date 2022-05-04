import numpy as np
import matplotlib.pyplot as plt
from lab2_tools import *
from prondict import *
import psutil
from timeit import time


def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    hmm1['transmat'][-1, -1] = 0
    K = hmm1['transmat'].shape[0] + hmm2['transmat'].shape[0] - 1
    concHMM = np.zeros((K, K))
    concprob = np.zeros(len(hmm1['startprob']) + len(hmm2['startprob'])-1)

    concHMM[: hmm1['transmat'].shape[0] - 1, : hmm1['transmat'].shape[1] - 1] = hmm1['transmat'][:-1, : -1]
    concHMM[hmm1['transmat'].shape[0] - 1:, hmm1['transmat'].shape[1] - 1:] = hmm2['transmat']
    concHMM[: hmm1['transmat'].shape[0] - 1, hmm1['transmat'].shape[1] - 1:] = np.outer(hmm1['transmat'][:-1, -1], hmm2['startprob'])

    concprob[:len(hmm1['startprob']) - 1] = hmm1['startprob'][: -1]
    concprob[len(hmm1['startprob'])-1:] = hmm1['startprob'][-1] * hmm2['startprob']

    concMeans = np.concatenate((hmm1['means'], hmm2['means']))
    conCovar = np.concatenate((hmm1['covars'], hmm2['covars']))

    return {"transmat": concHMM, "startprob": concprob, "means": concMeans, "covars": conCovar}



# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name.
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]

    for idx in range(1, len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """


def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    N, M = np.shape(log_emlik)
    forward_prob = np.zeros((N, M))
    for timestep in range(0, N):
        for state in range(0, M):
            if timestep == 0:
                forward_prob[0][state] = log_startprob[state] + log_emlik[0][state]
            else:
                forward_prob[timestep][state] = logsumexp(np.add(forward_prob[timestep-1, :], log_transmat[:M, state])) + log_emlik[timestep][state]

    return forward_prob


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N, M = np.shape(log_emlik)
    backward_prob = np.zeros((N, M))

    for i in range(N-2, -1, -1):
        for j in range(0, M):
            # backward_prob[i, j] = logsumexp(log_transmat[j,:M] + log_emlik[i+1, j] + backward_prob[i+1, :])
            backward_prob[i, j] = logsumexp(log_transmat[j,:-1] + log_emlik[i+1, :] + backward_prob[i+1, :])
    return backward_prob


def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N, M = np.shape(log_emlik)

    V = np.zeros((N, M))
    B = np.zeros((N, M), dtype=int)
    #print("LOG_EMLIK: ", log_emlik.shape, "  log_startp: ", log_startprob.shape, "   log_transm: ", log_transmat.shape)

    # initialize
    for i in range(M):
        V[0, i] = log_startprob[i] + log_emlik[0, i]

    for t in range(1, N):
        for s in range(0, M):
            V[t, s] = max(V[t-1, :M] + log_transmat[:M, s] + log_emlik[t, s])
            B[t, s] = np.argmax(V[t-1, :M] + log_transmat[:M, s])


    viterbi_loglik = np.max(V[-1, :])

    idx = int(np.argmax(V[-1, :])) # idx att börja på i B

    viterbi_path = []

    for i in range(len(log_emlik)-1, 0, -1):
        viterbi_path.insert(0, (i, B[i, idx]))
        idx = B[i, idx]
    viterbi_path.insert(0, (0, B[0, int(np.argmax(V[0, :]))]))

    return viterbi_loglik, np.array(viterbi_path)




def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

    # N, M = np.size(log_alpha)
    # log_gamma = np.zeros((N, M))
    #
    # for n in range(N):
    #     for i in range(M):
    #         log_gamma[n, i] =log_alpha[n, i] + log_beta[n, i]- logsumexp(log_alpha[N-1, i])

    return log_alpha + log_beta - logsumexp(log_alpha[len(log_alpha) - 1])




def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
#    N, D = np.shape(X)
#    N, M = np.shape(log_gamma)
#    print(X.shape, log_gamma.shape)
#    means = np.zeros((M, D))
#    covars = np.zeros((M, D))
#
#    for i in range(M):
#        means[i] = np.sum(log_gamma[:, i].reshape(-1, 1) * X, axis=0) / np.sum(log_gamma[:, i])
#        covars[i] = np.sum(log_gamma[:, i].reshape(-1, 1) * np.square(X-means[i])) / np.sum(log_gamma[:, i])
#        covars[i, covars < varianceFloor] = varianceFloor

### VÅR ####
    gamma = np.exp(log_gamma)
    means = np.zeros((log_gamma.shape[1], X.shape[1]))
    covars = np.zeros(means.shape)

    for i in range(means.shape[0]):
        gamma_sum = np.sum(gamma[:,i])
        means[i] = np.sum(gamma[:,i].reshape(-1, 1) * X, axis=0) / gamma_sum
        covars[i] = np.sum(gamma[:, i].reshape(-1, 1) * (X - means[i])**2, axis = 0) / gamma_sum
        covars[i, covars[i] < varianceFloor] = varianceFloor

    return means, covars


def compute_likelihood(alpha):
    return logsumexp(alpha[-1])


def likelihood_total(data, isolated, phoneHMMs):

    #t1 = time.perf_counter()
    wordHMMs = {}
    for digit in isolated.keys():
        wordHMMs[digit] = concatHMMs(phoneHMMs, isolated[digit])
    max_likelihood = None
    best_model = {}
    accuracy = 0
    for idx, x in enumerate(data):
        if x['speaker'] != 'ew':
            continue
        max_likelihood = -np.inf
        for digit in wordHMMs.keys():
            obsloglik = log_multivariate_normal_density_diag(x['lmfcc'], wordHMMs[digit]['means'],
                                                         wordHMMs[digit]['covars'])

            forwardP = forward(obsloglik, np.log(wordHMMs[digit]['startprob']), np.log(wordHMMs[digit]['transmat']))
            seq_likelihood = logsumexp(forwardP[-1])
            if seq_likelihood > max_likelihood:
                max_likelihood = seq_likelihood
                best_model[idx] = digit
        if x['digit'] == best_model[idx]:
            accuracy += 1
        #else:
        #    print("True digit: {}".format(x['digit']), "  HMM digit: {}".format(best_model[idx]))
    #t2 = time.perf_counter()
    #print("FORWARD TIME: ", t2-t1)
        print("The best model for utterance " + str(idx) + " was hmm: " + str(best_model[idx]))
        print("The real digit of utterance " + str(idx) + " was digit: " + str(x['digit']) + "\n")

    print("The accuracy of the predictions has been: " + str(np.round(accuracy / len(data) * 100, 2)/50) + "%")

def test_viterbi(data, phoneHMMs):
    t1 = time.perf_counter()
    wordHMMs = {}
    for digit in isolated.keys():
       wordHMMs[digit] = concatHMMs(phoneHMMs, isolated[digit])
    best_model = {}
    accuracy = 0

    for idx, x in enumerate(data):
        #if x['gender'] != 'woman'
       max_likelihood = -np.inf

       for digit in wordHMMs.keys():

           obsloglik = log_multivariate_normal_density_diag(x['lmfcc'], wordHMMs[digit]['means'], wordHMMs[digit]['covars'])
           vi_loglik, path = viterbi(obsloglik,  np.log(wordHMMs[digit]['startprob']), np.log(wordHMMs[digit]['transmat']))

           seq_likelihood = vi_loglik
           if seq_likelihood > max_likelihood:
               max_likelihood = seq_likelihood
               best_model[idx] = digit

       if x['digit'] == best_model[idx]:
           accuracy += 1
    t2 = time.perf_counter()
    print("VITERBI TIME: ", t2 - t1)

    #   print("The best model for utterance " + str(idx) + " was hmm: " + str(best_model[idx]))
    #   print("The real digit of utterance " + str(idx) + " was digit: " + str(x['digit']) + "\n")
    #print("The accuracy of the predictions has been: " + str(np.round(accuracy / len(data) * 100, 2)) + "%")




#### LOAD DATA ####

data = np.load('lab2_data.npz', allow_pickle=True)['data']
example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()

#phoneHMMs = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']

#### 5.1 HMM Likelihood and Recognition #####
def five_1():
    wordHMMs = {}
    wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])

    lpr_o = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs['o']['means'], wordHMMs['o']['covars'])



    plt.pcolormesh(lpr_o)
    plt.show()


##### 5.2 Forward Algorithm #####
def five_2():
    wordHMMs = {}
    wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    log_alpha = np.log(wordHMMs['o']['transmat'])
    log_pi = np.log(wordHMMs['o']['startprob'])

    log_gmm = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs['o']['means'], wordHMMs['o']['covars'])
    #gmm_forward = forward(log_gmm, log_pi, log_alpha)

    forwardP = forward(example['obsloglik'], log_pi, log_alpha)
    loglik = compute_likelihood(forwardP)

    """Convert the formula you have derived into log domain. Verify that the log likelihood ob-
     tained this way using the model parameters in wordHMMs['o'] and the observation sequence
      in example['lmfcc'] is the same as example['loglik']."""


    """
    Using your formula, score all the 44 utterances in the data array with each of the 11 HMM models in wordHMMs.
    """
    likelihood_total(data, isolated, phoneHMMs)
    return forwardP, log_gmm


##### 5.3 Viterbi Approximation #####
def five_3():
    wordHMMs = {}
    wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])

    log_alpha = np.log(wordHMMs['o']['transmat'])
    log_pi = np.log(wordHMMs['o']['startprob'])
    forwardP = forward(example['obsloglik'], log_pi, log_alpha)
    loglik = compute_likelihood(forwardP)
    viterbi_loglik, best_path = viterbi(example['obsloglik'], log_pi, log_alpha)
    plt.pcolormesh(forwardP.T)

    plt.plot(best_path[:, 0], best_path[:, 1],"r")
    plt.show()

    print(psutil.cpu_percent())

    test_viterbi(data, phoneHMMs)


##### 5.4 Backward Algorithm #####
def five_4():
    wordHMMs = {}
    wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    log_a = np.log(wordHMMs['o']['transmat'])
    log_pi = np.log(wordHMMs['o']['startprob'])
    example1 = example['obsloglik']

    log_lik = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs['o']['means'], wordHMMs['o']['covars'])
    example2 = log_lik

    gmm_backward = backward(log_lik, log_pi, log_a)
    backwardP = backward(example['obsloglik'], log_pi, log_a)
#    plt.pcolormesh(backwardP)
#    plt.show()
#    plt.pcolormesh(example['logbeta'])
#    plt.title('correct')
#    plt.show()
    return backwardP, example2, gmm_backward


def six_1():
    wordHMMs = {}
    wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    log_alpha, log_gmm_alpha = five_2()
    log_beta, log_lik, gmm_beta = five_4()
    posten = statePosteriors(log_alpha, log_beta)
    log_gmm = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs['o']['means'], wordHMMs['o']['covars'])
    gmm_posten = statePosteriors(log_gmm, log_gmm)


    linear_HMM_Post = np.exp(posten)
    linear_GMM_Post = np.exp(log_gmm_alpha)

    print('log domain HMM Posterior', posten)
    print('log domain GMM Posterios', gmm_posten)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    print(posten.shape)
    ax1.pcolormesh(posten)
    ax1.set_title('HMM posteriors')
    ax2.pcolormesh(gmm_posten)
    ax2.set_title('GMM posteriors')
    plt.show()


    print("SUM LINEAR_HMM_POST OVER TIMESTEPS (ROWS): \n", np.sum(linear_HMM_Post, axis=1), "\n\n")

    print("SUM LINEAR_HMM_POST OVER TIMESTEPS (ROWS) AND STATES (COLS): \n", np.sum(linear_HMM_Post))
    print("SUM LINEAR_GMM_POST OVER TIMESTEPS (ROWS) AND STATES (COLS): \n", np.sum(linear_GMM_Post))


def six_2(utterance, data_lmfcc):
    wordHMMs = {}
    for digit in isolated.keys():
        wordHMMs[digit] = concatHMMs(phoneHMMs, isolated[digit])

    wordHMMs["{}".format(utterance)] = concatHMMs(phoneHMMs, ['sil'] + prondict[utterance] + ['sil'])

    max_iterations = 20
    old_like = None


    iterations = []
    for iteration in range(max_iterations):
        log_gmm = log_multivariate_normal_density_diag(data_lmfcc,  wordHMMs[utterance]['means'], wordHMMs[utterance]['covars'])

        alpha_prob = forward(log_gmm, np.log(wordHMMs[utterance]['startprob']), np.log(wordHMMs[utterance]['transmat']))
        beta_prob = backward(log_gmm, np.log(wordHMMs[utterance]['startprob']), np.log(wordHMMs[utterance]['transmat']))
        state_posteriors = statePosteriors(alpha_prob, beta_prob)
        wordHMMs[utterance]['means'], wordHMMs[utterance]['covars'] = updateMeanAndVar(data_lmfcc, state_posteriors)

        log_likelihood_increase = compute_likelihood(alpha_prob)

        if (old_like is None) or (abs(log_likelihood_increase-old_like) > 1):
            old_like = log_likelihood_increase
            iterations.append(log_likelihood_increase)
        else:
            break
    return iterations

five_2()


#for i in prondict.keys():
#    it = six_2(i, data[10]['lmfcc'])
#    proc = np.exp(it)
#    print("Iterations and likelihood for utterance {}: {}".format(i, len(it) + 1), it)
