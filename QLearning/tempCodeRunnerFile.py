cmap = sns.cubehelix_palette(start = 2.5, rot = 1, gamma=0.7, as_cmap = True)
sns.heatmap(A, cmap=cmap)