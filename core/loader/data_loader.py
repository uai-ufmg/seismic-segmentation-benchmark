import numpy as np
import segyio
import os
import json
from torch.utils.data import Dataset


class SeismicDataset(Dataset):

    def __init__(self, data_path, labels_path, orientation, compute_weights=False, faulty_slices_list=None):
        self.data, self.labels = self.__load_data(data_path, labels_path)
        self.orientation = orientation

        # Removing faulty slices from the data if specified
        if faulty_slices_list is not None:
            self.__remove_faulty_slices(faulty_slices_list)
        
        self.n_inlines, self.n_crosslines, self.n_time_slices = self.data.shape
        
        self.n_classes = self.__process_class_labels()
        self.weights = self.__compute_class_weights() if compute_weights else None


    def __getitem__(self, index):
        if self.orientation == 'in':
            image = self.data[index, :, :]
            label = self.labels[index, :, :]
        else:
            image = self.data[:, index, :]
            label = self.labels[:, index, :]
        
        # Reshaping to 3D image
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        return image, label


    def __len__(self):
        return self.n_inlines if self.orientation == 'in' else self.n_crosslines
    
    
    def __load_data(self, data_path, labels_path):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f'File {data_path} does not exist.')
        
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f'File {labels_path} does not exist.')
        
        _, data_extension = os.path.splitext(data_path)
        
        # Loading data
        if data_extension in ['.segy', '.sgy']:
            inlines = []
        
            with segyio.open(data_path, 'r') as segyfile:
                segyfile.mmap()

                for inline in segyfile.ilines:
                    inlines.append(segyfile.iline[inline])

            data = np.array(inlines)
        else:
            data = np.load(data_path)

        # Loading labels
        labels = np.load(labels_path)
        
        return data, labels


    def __compute_class_weights(self):
        total_n_values = self.n_inlines * self.n_crosslines * self.n_time_slices
        # Weights are inversely proportional to the frequency of the classes in the training set
        _, counts = np.unique(self.labels, return_counts=True)
        
        return total_n_values / (counts*self.n_classes)
    

    def __remove_faulty_slices(self, faulty_slices_list):
        try:
            with open(faulty_slices_list, 'r') as json_buffer:
                # File containing the list of slices to delete
                faulty_slices = json.loads(json_buffer.read())

                self.data = np.delete(self.data, obj=faulty_slices['inlines'], axis=0)
                self.data = np.delete(self.data, obj=faulty_slices['crosslines'], axis=1)
                self.data = np.delete(self.data, obj=faulty_slices['time_slices'], axis=2)

                self.labels = np.delete(self.labels, obj=faulty_slices['inlines'], axis=0)
                self.labels = np.delete(self.labels, obj=faulty_slices['crosslines'], axis=1)
                self.labels = np.delete(self.labels, obj=faulty_slices['time_slices'], axis=2)

        except FileNotFoundError:
            print('Could not open the .json file containing the faulty slices.')
            print('Training with the whole volume instead.\n')

            pass
    

    def __process_class_labels(self):
        # Labels must be in the range [0, number_of_classes) for the loss function to work properly
        label_values = np.unique(self.labels)
        new_labels_dict = {label_values[i]: i for i in range(len(label_values))}

        for key, value in zip(new_labels_dict.keys(), new_labels_dict.values()):
            self.labels[self.labels == key] = value
        
        return len(label_values)
    

    def get_class_weights(self):
        return self.weights
    

    def get_n_classes(self):
        return self.n_classes
