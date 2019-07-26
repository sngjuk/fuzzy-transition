import numpy as np
from scipy.spatial import distance

glossary = None
path = None
model = None


def dfs(cur_name, target_name, cur_path, cur_prob, depth):
    if len(cur_prob) and abs(cur_prob[-1]) < 0.01 or depth > 8:
        return

    cur_path = cur_path.copy()

    if cur_name == target_name:
        # return [[path, probability], ... ]
        path.append([cur_path, cur_prob])
        return

    reason_dict = glossary[cur_name].reason

    for reason in reason_dict:
        # prevent cycle
        if reason in cur_path:
            continue
        dfs(reason, target_name, cur_path + [reason], cur_prob + [reason_dict[reason][1]], depth+1)


def search_path(gs, gsv, model_in, source, dest):
    global glossary
    global glossary_vector
    global path
    global model

    glossary = gs
    glossary_vector = gsv
    path = []
    model = model_in

    usr_input = source, dest
    source_sim = None
    dest_sim = None

    # if source and dest name is not exist in glossary
    if source not in glossary:
        source, source_sim = most_sim_name(list(glossary.keys()), glossary_vector, model.get_word_vector(source))
    if dest not in glossary:
        dest, dest_sim = most_sim_name(list(glossary.keys()), glossary_vector, model.get_word_vector(dest))

    dfs(source, dest, [source], [], depth=0)

    # concat source and dest
    for idx, i in enumerate(path):
        if not source == usr_input[0]:
            path[idx][0].insert(0, usr_input[0])
            path[idx][1].insert(0, source_sim)

    for idx, i in enumerate(path):
        if not dest == usr_input[1]:
            path[idx][0].append(usr_input[1])
            path[idx][1].append(dest_sim)

    return path


def most_sim_name(name_list, near_vectors, cur_vector):

    distances = distance.cdist(np.array([cur_vector]), np.array(near_vectors), "cosine")[0]
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    return name_list[int(min_index)], 1 - min_distance


def hop_vector_space(cur_name, target_name, target_vector, cur_path, cur_prob):
    if cur_name == target_name:
        # return [[path, probability], ... ]
        path.append([cur_path, cur_prob])
        return

    skipped_result = model.filtered_nearest_neighbor(cur_name, 150, 0.55)
    near_vectors = [model.get_word_vector(i[0]) for i in skipped_result]

    next_name, similarity = most_sim_name(skipped_result, near_vectors, target_vector)

    print(f'/ {cur_name} -> {next_name}, sim: {similarity}')

    if next_name in cur_path:
        # return [[path, probability], ... ]
        del cur_path[-1]
        del cur_prob[-1]
        path.append([cur_path + [target_name],
                     cur_prob + [distance.cosine(model.get_word_vector(cur_path[-1]), target_vector) ]])
        print('fall into local minimum; duplicated loop! -> terminate!')
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


def dfs_with_length(cur_name, cur_path, cur_prob, depth, length):
    if depth >= length:
        path.append([cur_path, cur_prob])
        return

    reason_dict = glossary[cur_name].reason

    for reason in reason_dict:
        # prevent cycle
        if reason in cur_path:
            continue
        dfs_with_length(reason, cur_path + [reason], cur_prob + [reason_dict[reason][1]], depth+1, length)


def search_hidden_path_with_length(gs, gsv, model_in, source, length=4):
    global glossary
    global glossary_vector
    global path
    global model

    glossary = gs
    glossary_vector = gsv
    path = []
    model = model_in

    usr_source = source
    source_sim = None

    # if source name is not exist in glossary
    if source not in glossary:
        source, source_sim = most_sim_name(list(glossary.keys()), glossary_vector, model.get_word_vector(source))

    dfs_with_length(source, [source], [], 0, length)

    # concat source
    for idx, i in enumerate(path):
        if not source == usr_source:
            path[idx][0].insert(0, usr_source)
            path[idx][1].insert(0, source_sim)

    return path
