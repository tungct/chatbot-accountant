import pickle

class Io_Utils():
    def save_pickle(self, data, path):
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path):
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        return data