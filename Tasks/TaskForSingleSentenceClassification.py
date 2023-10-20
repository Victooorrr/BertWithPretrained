import sys

sys.path.append('../')
from model import BertForSentenceClassification
from model import BertConfig
from utils import LoadSingleSentenceClassificationDataset
from utils import logger_init
from transformers import BertTokenizer
import logging
import torch
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'Aid')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'val.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'test.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = '_!_'
        self.is_sample_shuffle = True
        self.batch_size = 12
        self.max_sen_len = None
        self.num_labels = 8
        self.epochs = 10
        self.learning_rate = 1e-5
        self.model_val_per_epoch = 2
        logger_init(log_file_name='single', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")


def train(config):
    model = BertForSentenceClassification(config,
                                          config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=bert_tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    max_acc = 0
    # output_file = open('incorrect_predictions.txt', 'w', encoding='utf-8')
    val_accuracy_history = []
    test_accuracy_history = []
    train_loss_history = []

    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            #
            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = model(
                input_ids=sample,
                attention_mask=padding_mask,
                token_type_ids=None,
                position_ids=None,
                labels=label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == label).float().mean()

            # incorrect_preds = (logits.argmax(1) != label).nonzero()  # Find indices of incorrect predictions
            # for i in incorrect_preds:
            #     incorrect_idx = i.item()
            #     incorrect_ids = sample[:, incorrect_idx].tolist()  # Get the token IDs of the incorrect sample as a list
            #     # Convert token IDs back to text
            #     incorrect_text = tokenizer.decode(incorrect_ids, skip_special_tokens=True)
            #     correct_label = label[incorrect_idx].item()  # Get the correct label
            #     predicted_label = logits[incorrect_idx].argmax().item()  # Get the predicted label
            #     # Write the incorrect prediction to the file
            #     output_file.write(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}],Correct Label: {correct_label}, Predicted Label: {predicted_label} "
            #                       f"Incorrect Sample: {incorrect_text}\n")
            if idx % 10 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")


        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        # if (epoch + 1) % config.model_val_per_epoch == 0:
        # 记录数据
        val_accuracy = evaluate(val_iter, model, config.device, data_loader.PAD_IDX)
        val_accuracy_history.append(val_accuracy)

        test_accuracy = evaluate(test_iter, model, config.device, data_loader.PAD_IDX)
        test_accuracy_history.append(test_accuracy)


        train_loss_history.append(train_loss)
        logging.info(f"Accuracy on val {val_accuracy:.3f}")
        if val_accuracy > max_acc:
            max_acc = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"## Save model on epoch {epoch + 1}##")
    plt.figure(1)
    plt.plot(range(1, config.epochs + 1), val_accuracy_history, marker='o', linestyle='-')
    plt.plot(range(1, config.epochs + 1), test_accuracy_history, marker='o', linestyle='-', color='g')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('./accuracy_plot.png')

    plt.figure(2)
    plt.plot(range(1, config.epochs + 1), train_loss_history, marker='o', linestyle='-')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('./train_loss_plot.png')


def inference(config):
    model = BertForSentenceClassification(config,
                                          config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行预测......")
    model = model.to(config.device)
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=BertTokenizer.from_pretrained(
                                                              config.pretrained_model_dir).tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    acc, predicted_labels, true_labels = evaluate4test(test_iter, model, device=config.device, PAD_IDX=data_loader.PAD_IDX)
    logging.info(f"Acc on test:{acc:.3f}")
    class_names = ["Non-topical", "Question", "Statement", "Support", "Conflict", "Clarify", "Summary", "Comment"]
    plot_confusion_matrix(true_labels, predicted_labels, class_names)


def evaluate(data_iter, model, device, PAD_IDX):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            padding_mask = (x == PAD_IDX).transpose(0, 1)
            logits = model(x, attention_mask=padding_mask)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        model.train()
        return acc_sum / n


def evaluate4test(data_iter, model, device, PAD_IDX):
    true_labels = []  # 真实标签
    predicted_labels = []  # 预测标签
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            padding_mask = (x == PAD_IDX).transpose(0, 1)
            logits = model(x, attention_mask=padding_mask)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
            predicted_labels.extend(logits.argmax(1).tolist())
            true_labels.extend(y.tolist())
        model.train()
    return acc_sum / n, predicted_labels, true_labels



def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('./confusion_matrix.png')



if __name__ == '__main__':
    import os
    directory = "../data/Aid"
    files_to_delete = ['train_None.pt', 'val_None.pt', 'test_None.pt']
    for file in files_to_delete:
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")
        else:
            print(f"{file_path} does not exist")

    if os.path.exists("../cache/model.pt"):
        os.remove("../cache/model.pt")
        print(f"Deleted model.pt")
    else:
        print(f"model.pt does not exist")
    model_config = ModelConfig()
    train(model_config)
    inference(model_config)

