import torch
import numpy as np

# Variables for file locations.
TRAIN_X_FN = "UCI HAR Dataset/train/X_train.txt"
TRAIN_Y_FN = "UCI HAR Dataset/train/y_train.txt"
TRAIN_SUBJ_FN = "UCI HAR Dataset/train/subject_train.txt"
TEST_X_FN = "UCI HAR Dataset/test/X_test.txt"
TEST_Y_FN = "UCI HAR Dataset/test/y_test.txt"
TEST_SUBJ_FN = "UCI HAR Dataset/test/subject_test.txt"

# Variables for settings.
SEQ_LEN = 10
INPUT_SIZE = 561
OUTPUT_SIZE = 5

# Create a class to hold each record.
class Record:
    def __init__(self, user_id, X, y):
        self.id = torch.tensor(user_id)
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

# Create a class to hold all the records.
from torch.utils.data import Dataset
import torch.nn.functional as F
class ActivityDataLoader(Dataset):
    def __init__(self, seq_len, id_list=None, x_list=None, y_list=None):
        self.seq_len = seq_len
        # Create an empty list to hold records.
        self.records = []
        # If an lists are provided and all the same size, instantiate the records.
        if id_list != None and x_list != None and y_list != None:
            if len(id_list)==len(y_list) and len(y_list)==len(x_list):
                # Create a record for each line.
                for id, x, y in zip(id_list, x_list, y_list):
                    self.add_new_record(id, x, y)
    # Methods to add records.
    def add_new_record(self, id, x, y):
        self.records.append(Record(id, x, y))
    def append(self, record):
        self.records.append(record)
    # Method to get length (required).
    def __len__(self):
        return len(self.records) - self.seq_len + 1
    # Method to get item (required).
    def __getitem__(self, index):
        records = self.records[index:index+self.seq_len]
        # Make X a 2D tensor (sequence length x input features).
        X = torch.tensor(np.array([record.X.numpy() for record in records]))
        # One-hot encode the classification, such that 1 --> [1, 0, 0, 0, 0], and so on.
        y = F.one_hot(records[-1].y-1, 5)
        return X, y

import torch.nn as nn
class ActivityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super(ActivityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.stack = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, 5),
            nn.Softmax()
        )
    def forward(self, X):
        out, _ = self.lstm(X)
        out = self.stack(out)
        return(out)

def main():
    # Load all files into the program.
    with open(TRAIN_X_FN, 'r') as f:
        train_x_contents = f.readlines()
        train_x_contents = [[float(x) for x in line.split()] for line in train_x_contents]
    with open(TRAIN_Y_FN, 'r') as f:
        train_y_contents = f.readlines()
        train_y_contents = [int(y) for y in train_y_contents]
    with open(TRAIN_SUBJ_FN, 'r') as f:
        train_subj_contents = f.readlines()
        train_subj_contents = [int(s) for s in train_subj_contents]
    with open(TEST_X_FN, 'r') as f:
        test_x_contents = f.readlines()
        test_x_contents = [[float(x) for x in line.split()] for line in test_x_contents]
    with open(TEST_Y_FN, 'r') as f:
        test_y_contents = f.readlines()
        test_y_contents = [int(y) for y in test_y_contents]
    with open(TEST_SUBJ_FN, 'r') as f:
        test_subj_contents = f.readlines()
        test_subj_contents = [int(s) for s in test_subj_contents]

    # Create dataset objects.
    train_ds = ActivityDataLoader(SEQ_LEN, train_subj_contents, train_x_contents, train_y_contents)
    test_ds = ActivityDataLoader(SEQ_LEN, test_subj_contents, test_x_contents, test_y_contents)

    # Display some information.
    print(f"Number of training data points: {len(train_ds)}")
    print(f"Shape of feature: {test_ds[0][0].shape}")

if __name__ == "__main__":
    main()