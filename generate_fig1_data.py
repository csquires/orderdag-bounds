from generate_cpdags import save_dags, save_mec_info

densities = [.1, .2, .5, .7]
node_sizes = [3, 5, 10, 30, 50, 70, 90, 110]
ndags = 2000

for density in densities:
    for p in node_sizes:
        save_dags(p, density, ndags)

for density in densities:
    for p in node_sizes:
        save_mec_info(p, density, ndags, [1, 2])


