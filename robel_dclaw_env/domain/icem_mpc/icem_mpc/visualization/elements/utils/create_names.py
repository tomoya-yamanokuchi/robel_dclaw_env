



def fname(dataname, iter_outer_loop, iter_inner_loop, num_sample_i):
    return "{}_outer{}_inner{}_sample{}".format(dataname, iter_outer_loop, iter_inner_loop, num_sample_i)


def title(dataname, iter_outer_loop, iter_inner_loop, num_sample_i):
    return fname(dataname, iter_outer_loop, iter_inner_loop, num_sample_i) + ".png"
