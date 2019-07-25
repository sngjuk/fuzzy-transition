import re

import numpy as np
from scipy.spatial import distance

glossary = None
path = None
model = None


def dfs(cur_name, target_name, cur_path, cur_prob):
    cur_path = cur_path.copy()
    if cur_name == target_name:
        # return [[path, probability], ... ]
        path.append([cur_path, cur_prob])
        return

    imp_dict = glossary[cur_name].implication

    for implication in imp_dict:
        dfs(implication, target_name, cur_path + [implication], cur_prob + [imp_dict[implication][1]])


def search_path(gs, source, dest):
    global glossary
    global path

    path = []
    glossary = gs
    dfs(source, dest, [source], [])

    return path


def most_sim_name(cur_vector, near_vectors):

    distances = distance.cdist(np.array([cur_vector]), np.array(near_vectors), "cosine")[0]
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    return min_index, 1-min_distance


def skip_same_words(result, name, sim_threshold=0.55):
    ret = []
    for i in result:
        my_regex = r".*" + re.escape(name) + r".*"
        if re.search(my_regex, i[0]):
            continue
        elif i[1] > sim_threshold:
            ret.append(i[0])

    return ret


def hop_vector_space(cur_name, target_name, target_vector, cur_path, cur_prob):
    if cur_name == target_name:
        # return [[path, probability], ... ]
        path.append([cur_path, cur_prob])
        return

    result = model.nearest_words(cur_name, 150)
    skipped_result = skip_same_words(result, cur_name)
    near_vectors = [model.get_word_vector(i) for i in skipped_result]

    min_index, similarity = most_sim_name(target_vector, near_vectors)
    next_name = skipped_result[int(min_index)]

    print(f'/ {cur_name} -> {next_name}, sim: {similarity}')

    if next_name in cur_path:
        # return [[path, probability], ... ]
        del cur_path[-1]
        del cur_prob[-1]
        path.append([cur_path + [target_name],
                     cur_prob + [distance.cosine(model.get_word_vector(cur_path[-1]), target_vector) ]])
        print('duplicated loop! -> terminate!')
        return

    hop_vector_space(next_name, target_name, target_vector,
                     cur_path + [next_name], cur_prob + [similarity])


def across_vector_space(gs, model_in, source, dest):
    global glossary
    global path
    global model

    glossary = gs
    path = []
    model = model_in

    hop_vector_space(source, dest, model.get_word_vector(dest), [source], [])

    return path
