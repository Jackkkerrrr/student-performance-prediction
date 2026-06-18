# TODO: complete this file.
from neural_network import AutoEncoder, train, load_data
import numpy as np
import torch


def ensemble_predict(models, input_vec):
    preds = []
    if isinstance(input_vec, torch.Tensor):
        input_tensor = input_vec.detach().clone().unsqueeze(0).float()
    else:
        input_tensor = torch.tensor(input_vec).unsqueeze(0).float()
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(input_tensor).squeeze(0).cpu().numpy()
            pred = (output >= 0.5).astype(int)
            preds.append(pred)
    stacked = np.stack(preds, axis=0)  # [num_models, num_questions]
    mean_pred = np.mean(stacked, axis=0)
    majority_vote = (mean_pred >= 0.5).astype(int)
    return majority_vote  # shape: (num_questions,)

def evaluate(models, data):
    ensemble_preds = []
    for user_id in data['user_id']:
        input_vec = zero_train_matrix[user_id]
        pred = ensemble_predict(models, input_vec)
        ensemble_preds.append(pred)
    ensemble_preds = np.array(ensemble_preds)
    correct = 0
    total = 0
    for idx, user_id in enumerate(data['user_id']):
        qids = data['question_id'][idx]
        labels = data['is_correct'][idx]
        pred_labels = ensemble_preds[idx][qids]
        if isinstance(labels, int):
            total += 1
            correct += int(pred_labels == labels)
        else:
            total += len(labels)
            correct += np.sum(pred_labels == labels)
    acc = correct / total
    return acc

if __name__ == "__main__":
    # Load the data
    zero_train_matrix, train_matrix, valid_data, test_data = load_data("./data")
    num_models = 3
    num_question = train_matrix.shape[1]
    k = 50  # latent dimension
    lr = 0.01  # learning rate
    lamb = 0.001  # regularization parameter
    num_epoch = 30  # number of epochs
    autoencoders = []

    for i in range(num_models):
        # load the data
        idx = np.random.choice(train_matrix.shape[0], size=train_matrix.shape[0], replace=True)
        boot_train_matrix = train_matrix[idx]
        boot_zero_train_matrix = zero_train_matrix[idx]
        # initialize parameters
        model = AutoEncoder(num_question, k)
        # train the model
        train(model, lr, lamb, boot_train_matrix, boot_zero_train_matrix, valid_data, num_epoch)
        autoencoders.append(model)
        print(f"Model {i + 1} trained.")

    val_acc = evaluate(autoencoders, valid_data)
    test_acc = evaluate(autoencoders, test_data)

    print(f"Ensemble Validation Accuracy: {val_acc:.4f}")
    print(f"Ensemble Test Accuracy: {test_acc:.4f}")