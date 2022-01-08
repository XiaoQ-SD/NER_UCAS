import pickle

def load_model(file_name):
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return f

def save_model(model, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list