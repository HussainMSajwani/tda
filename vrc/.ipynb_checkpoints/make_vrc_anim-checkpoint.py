import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import tqdm
import os
import imageio

key = lambda filename: int(filename[6:][:-4])
sorted_listdir = lambda directory: sorted(os.listdir(directory), key=key)

d = lambda x0, y0, x1, y1: np.sqrt((x0-x1)**2 + (y0-y1)**2)

def get_points_within(eps, X, Y):
    points_in_range = [] 

    for x0, y0 in zip(X, Y):
        points_in_range.append([])
        for x1, y1 in zip(X, Y):
            if d(x0, y0, x1, y1) <= eps:
                points_in_range[-1].append((x1, y1))
    
    return points_in_range

def makeframe(eps, j):
    np.random.seed(123)
    n = 50
    fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    r = 1 + 0.1*np.random.randn(n)
    theta = 2*np.pi*np.random.random(n)

    X = r*np.cos(theta)
    Y = r*np.sin(theta)

    circles = [plt.Circle((x, y), eps, color='b', fill=True, alpha=0.3) for x, y in zip(X, Y)]
    points_in_range = get_points_within(eps, X, Y)

    for i, circle in enumerate(circles):
        axs.add_patch(circle)
        dim_of_simp = len(points_in_range[i]) 
        if dim_of_simp > 1:
            for couple in combinations(points_in_range[i], 2):
                X_ = [couple[0][0], couple[1][0]]
                Y_ = [couple[0][1], couple[1][1]]

                plt.plot(X_, Y_, color="red")
                #plt.plot(x_edge, y_edge)

    points_in_range

    plt.scatter(X, Y, color="black", s=300)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.savefig(f"vrc/anim/frame_{j}.png")
    plt.close()
    
for i, eps in tqdm.tqdm(enumerate(np.linspace(0, 0.8, 60))):
    makeframe(eps, i)
    
images = []
for filename in tqdm.tqdm(sorted_listdir("vrc/anim/")):
    images.append(imageio.imread("vrc/anim/" + filename))
imageio.mimsave('vrc/anim.gif', images, duration=0.1)