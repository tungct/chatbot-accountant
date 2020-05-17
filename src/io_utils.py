import pickle
import pandas as pd

class Io_Utils():
    def save_pickle(self, data, path):
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path):
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def load_xlsx(self, path):
        xl = pd.ExcelFile(path)
        xl = xl.parse(str(xl.sheet_names[0]))
        return xl

if __name__ == '__main__':
    xl = pd.ExcelFile('../data/chatbot-accountant-qa.xlsx')
    print(xl.sheet_names)
    xl = xl.parse('Trang tính1')
    print(len(xl))
    # dfs = pd.DataFrame(dfs, columns=dfs.keys())
    print(xl['stt'][0], xl['answer'][0])

