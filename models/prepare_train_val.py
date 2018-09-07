from models.dataset import data_path
from sklearn.model_selection import StratifiedKFold, KFold
import os



def get_split(fold):

    train_path = os.path.join(data_path, 'train','images')
    ids = next(os.walk(train_path))[2]

    #classes = pd.read_csv('classes.csv')
    #classes['class_type'] = classes['foreground'] + '_' + classes['background']
    #classes['file_idx'] = classes.filename.str[:-4]
    #class_types = classes.class_type[classes.file_idx.isin(ids)]
    #class_types = class_types.reset_index()
    #y = class_types['class_type']

    X = range(0,len(ids))
    #skf = StratifiedKFold(n_splits=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=47)

    #train_index, test_index  = list(skf.split(X, y))[fold]
    train_index, test_index  = list(kf.split(X))[fold]

    train_file_names = [ids[i][:-4] for i in train_index]
    val_file_names = [ids[i][:-4] for i in test_index]

    return train_file_names, val_file_names
