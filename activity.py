# Variables for file locations.
TRAIN_X_FN = "UCI HAR Dataset/train/X_train.txt"
TRAIN_Y_FN = "UCI HAR Dataset/train/y_train.txt"
TRAIN_SUBJ_FN = "UCI HAR Dataset/train/subject_train.txt"
TEST_X_FN = "UCI HAR Dataset/test/X_test.txt"
TEST_Y_FN = "UCI HAR Dataset/test/y_test.txt"
TEST_SUBJ_FN = "UCI HAR Dataset/test/subject_test.txt"

# Create a class to hold each record.
class Record:
    def __init__(self, user_id, X, y):
        self.id = user_id
        self.X = X
        self.y = y

# Create a class to hold all the records.
from torch.utils.data import Dataset
class ActivityDataset(Dataset):
    def __init__(self, id_list=None, x_list=None, y_list=None):
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
        return len(self.records)
    # Method to get item (required).
    def __getitem__(self, index):
        record = self.records[index]
        return record.X, record.y

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
    train_ds = ActivityDataset(train_subj_contents, train_x_contents, train_y_contents)
    test_ds = ActivityDataset(test_subj_contents, test_x_contents, test_y_contents)

    # Display some information.
    print(f"Number of training data points: {len(train_ds)}")
    print(f"Number of features: {len(test_ds[0][0])}")

if __name__ == "__main__":
    main()