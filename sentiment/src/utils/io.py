import pickle

def save_model(model, model_dir):
    with open(model_dir, "wb") as model_file:
        pickle.dump(model, model_file)
        model_file.close()


def load_model(model_dir):
    with open(model_dir, "rb") as model_file:
        model = pickle.load(model_file)
        model_file.close()
    return model