from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0

    for i, j, c in zip(data["user_id"], data["question_id"], data["is_correct"]):
        z = theta[i] - beta[j]
        log_lklihood += c * np.log(sigmoid(z)) + (1 - c) * np.log(1 - sigmoid(z))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_ids = data["user_id"]
    question_ids = data["question_id"]
    is_corrects = data["is_correct"]

    theta_grads = np.zeros(theta.shape[0])
    beta_grads = np.zeros(beta.shape[0])

    for i in set(user_ids):
        theta_grads[i] = np.sum([is_corrects[k] - sigmoid(theta[i] - beta[question_ids[k]])
                        for k in range(len(is_corrects)) if user_ids[k] == i])
    
    theta += lr * theta_grads

    for j in set(question_ids):
        beta_grads[j] = np.sum([- is_corrects[k] + sigmoid(theta[user_ids[k]] - beta[j])
                        for k in range(len(is_corrects)) if question_ids[k] == j])
    
    beta += lr * beta_grads

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros((max(data["user_id"]) + 1))
    beta = np.zeros((max(data["question_id"]) + 1))

    train_log_likes = []
    val_log_likes = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_log_likes.append(neg_lld)

        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_log_likes.append(val_neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)

        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_log_likes, val_log_likes


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 5e-3
    iters = 50

    print(f"Learning rate: {lr}")
    print(f"# of Iterations: {iters}")

    theta, beta, val_acc_lst, train_log_likes, val_log_likes = irt(train_data, val_data, lr, iters)

    averaged_train_log_likes = np.array(train_log_likes) / len(train_data["user_id"])
    averaged_val_log_likes = np.array(val_log_likes) / len(val_data["user_id"])

    plt.figure(figsize=(16, 9))
    plt.plot(averaged_train_log_likes, label="Averaged Training Negative Log-likelihoods")
    plt.plot(averaged_val_log_likes, label="Averaged Validation Negative Log-likelihoods")
    plt.xticks(range(1, iters + 1, 10))
    plt.title(f"Training and Validation Negative Log-likelihoods per Iteration\n(lr: {lr}, # of iter: {iters})")
    plt.xlabel("Iterations")
    plt.ylabel("Negative Log-likelihoods")
    plt.legend()
    plt.savefig("./graphs/train-val-log-likelihoods")

    final_val_acc = evaluate(val_data, theta, beta)
    final_test_acc = evaluate(test_data, theta, beta)

    print(f"Final validation accuracy: {final_val_acc}")
    print(f"Final test accuracy: {final_test_acc}")
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    j1 = 789
    j2 = 110
    j3 = 1536

    p1 = sigmoid(theta - beta[j1])
    p2 = sigmoid(theta - beta[j2])
    p3 = sigmoid(theta - beta[j3])

    plt.figure(figsize=(16, 9))
    plt.scatter(theta, p1, label=f"question {j1}")
    plt.scatter(theta, p2, label=f"question {j2}")
    plt.scatter(theta, p3, label=f"question {j3}")
    plt.title("Probability of the correct response for questions per theta")
    plt.xlabel("Theta")
    plt.ylabel("Probability of the correct response")
    plt.legend()
    plt.savefig("./graphs/correct-response-probs")
    
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
