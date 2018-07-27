import causaldag as cd
import save_utils
import numpy as np
import config
import os


def save_dags(n, s, ndags):
    run_folder = os.path.join(config.DATA_FOLDER, 's={s}/n={n}'.format(s=s, n=n))
    os.makedirs(run_folder)
    settings = {
        'nnodes': n,
        'sparsity': s,
        'ndags': ndags
    }
    save_utils.save_yaml(settings, os.path.join(run_folder, 'settings.yml'))

    dags = cd.random.directed_erdos(n, s, size=ndags)
    print('generated dags')
    cpdags = [d.cpdag for d in dags]
    print('computed cpdags')
    num_unoriented = [len(c.edges) for c in cpdags]
    save_utils.save_list(num_unoriented, os.path.join(run_folder, 'num_unoriented.txt'))

    for i, (d, c) in enumerate(zip(dags, cpdags)):
        dag_folder = os.path.join(run_folder, 'dag%d' % i)
        os.makedirs(dag_folder)
        dag_amat, node_list = d.to_amat()
        cpdag_amat, _ = c.to_amat(node_list)
        np.save(os.path.join(dag_folder, 'dag.npy'), dag_amat)
        np.save(os.path.join(dag_folder, 'cpdag.npy'), cpdag_amat)


if __name__ == '__main__':
    np.random.seed(1729)
    s = .5
    m = 1000
    # for i in range(10):
    #     save_dags((i+1)*10, s, m, i)
    import matplotlib.pyplot as plt

    # num_unoriented_mat = np.zeros((10, m))
    # for i in range(10):
    #     num_unoriented_mat[i] = save_utils.load_list(os.path.join(config.DATA_FOLDER, 'run%d' % i, 'num_unoriented.txt'))

    num_unoriented_mat = np.zeros((10, 10))
    for i in range(1, 11):
        fn = '/Users/chandlersquires/Desktop/Rorderdagbounds/numundirected_n=%d.txt' % (i * 10)
        a = open(fn).read().split()
        a = list(map(int, a))
        num_unoriented_mat[i-1] = a

    avg_num_unoriented = np.mean(num_unoriented_mat, axis=1)
    std_num_unoriented = np.std(num_unoriented_mat, axis=1)
    total_num_zero = (num_unoriented_mat == 0).sum(axis=1)

    plt.plot(avg_num_unoriented)
    plt.ion()
    plt.show()







