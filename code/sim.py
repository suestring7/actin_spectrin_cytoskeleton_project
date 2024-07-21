import os, sys, re, cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import glob
import scipy
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import matplotlib.animation as animation
import numpy as np
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from collections import Counter
from matplotlib.pyplot import *
import itertools

from utils import *


def simRandD(box=100, d=5, n=20):
    points = []
    i = 0
    while i < n:
        t_vx = np.random.rand(2) * box
        t_flag = 1
        for vx in points:
            if np.linalg.norm(t_vx - vx) < d:
                t_flag = 0
                break
        if t_flag == 1:
            points.append(t_vx)
            i += 1
    return np.array(points)


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)

    is_out = np.zeros([len(points), 1])
    for i, j in edges:
        is_out[i] = 1
        is_out[j] = 1
    b_vxs = np.array([points[i] for i in range(len(points)) if is_out[i] == 1])

    return edges, b_vxs


def calSpForce(length, Sp_L=0, Sp_coeff=10):
    if length > Sp_L:
        return -(length - Sp_L) * Sp_coeff
    else:
        return 0


def getAcc(pos, link_list):
    acc = np.zeros(pos.shape)
    link_vec = pos[link_list[:, 0], :] - pos[link_list[:, 1], :]

    link_L = np.linalg.norm(link_vec, axis=1)
    force = np.array([calSpForce(x) for x in link_L])
    ax = force * link_vec[:, 0] / link_L
    ay = force * link_vec[:, 1] / link_L
    np.add.at(acc[:, 0], link_list[:, 0], ax)
    np.add.at(acc[:, 1], link_list[:, 0], ay)
    np.add.at(acc[:, 0], link_list[:, 1], -ax)
    np.add.at(acc[:, 1], link_list[:, 1], -ay)
    return acc


def getVel(pos, link_list):
    vel = np.zeros(pos.shape)
    link_vec = pos[link_list[:, 0], :] - pos[link_list[:, 1], :]

    link_L = np.linalg.norm(link_vec, axis=1)
    force = np.array([calSpForce(x) for x in link_L])
    vx = force * link_vec[:, 0] / link_L
    vy = force * link_vec[:, 1] / link_L
    np.add.at(vel[:, 0], link_list[:, 0], vx)
    np.add.at(vel[:, 1], link_list[:, 0], vy)
    np.add.at(vel[:, 0], link_list[:, 1], -vx)
    np.add.at(vel[:, 1], link_list[:, 1], -vy)
    return vel


def sim2D(size=128, space=6):
    nx = ny = int(size / space)
    return np.array(
        [
            [((x * 2 - y % 2) * space) + 10, (y * np.sqrt(3) * space)]
            for x in range(nx)
            for y in range(ny)
        ]
    )


def convertMatrix2List(links):
    n_vx = len(links)
    link_list = []
    for i in range(n_vx):
        for j in range(i + 1, n_vx):
            if links[i][j] == 1:
                link_list.append([i, j])
    return np.array(link_list)


def generateLinks(vxs, img=None, thre=[0, 3], plot=1):
    # first step
    n_vx = len(vxs)
    tri = Delaunay(vxs)
    small_tri = [
        x
        for x in tri.simplices
        if thre[0] <= np.linalg.norm(vxs[x[0]] - vxs[x[1]]) <= thre[1]
        and thre[0] <= np.linalg.norm(vxs[x[2]] - vxs[x[1]]) <= thre[1]
        and thre[0] <= np.linalg.norm(vxs[x[0]] - vxs[x[2]]) <= thre[1]
    ]
    links = np.zeros([n_vx, n_vx])
    for x in small_tri:
        links[x[0]][x[1]] = links[x[1]][x[0]] = links[x[0]][x[2]] = links[x[2]][
            x[0]
        ] = links[x[1]][x[2]] = links[x[2]][x[1]] = 1
    if plot:
        plt.figure(figsize=[10, 10])
        if img:
            plt.imshow(img)
        if small_tri:
            plt.triplot(vxs[:, 0], vxs[:, 1], small_tri, c="g")
        plt.plot(
            vxs[:, 0],
            vxs[:, 1],
            ".",
            markerfacecolor="purple",
            markersize=10,
        )
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
    return links


# vxs = simRandD(box=64*16, d=100, n=70)

# test_2d = glob.glob("../data/rendered/soma/2D/*.txt")
# vxs = 16 * readVXSfromLoc(test_2d[1], plot=1)
# links = generateLinks(vxs, thre=[50, 280])

em = cv2.imread("../data/em/2.png", -1)
gt = cv2.imread("../data/em/2_marked.png", -1)
# plt.imshow(gt[:, :, 2] - gt[:, :, 1])
marker = gt[:, :, 2] - gt[:, :, 1]
contours = cv2.findContours(
    marker.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)[0]
contour_centers = np.zeros([len(contours) - 1, 2])
for i in range(len(contours) - 1):
    contour_centers[i] = contours[i + 1].reshape([-1, 2]).mean(axis=0)

vxs = contour_centers
n_vx = len(vxs)
gt_sp = cv2.imread("../data/em/2_marked_final.png", -1)
# plt.figure(figsize=[10, 10])
# plt.imshow(em)
marker_sp = gt_sp[:, :, 1] - gt_sp[:, :, 0]
links = np.zeros([n_vx, n_vx])
close_mid = np.zeros(marker_sp.T.shape)
for i in range(n_vx):
    close_mid[vxs[i][0].astype(int), vxs[i][1].astype(int)] = 1

close_mid = fftconvolve(close_mid, np.ones([10, 10]), mode="same")

# plt.figure(figsize=[10,10])
# plt.imshow(close_mid.T)
dist = distance_matrix(vxs, vxs)
for i in range(n_vx):
    for j in range(i + 1, n_vx):
        if dist[i][j] < 110:
            if (
                sum_line_cnct(marker_sp.T, vxs[i].astype(int), vxs[j].astype(int))
                >= dist[i][j] * 0.7
            ):
                if (
                    sum_line_cnt(close_mid, vxs[i].astype(int), vxs[j].astype(int))[0]
                    < 3
                ):
                    links[i, j] = links[j][i] = 1
                    plt.plot([vxs[i][0], vxs[j][0]], [vxs[i][1], vxs[j][1]], color="r")
                # else:
                #    plt.plot([vxs[i][0],vxs[j+1][0]],[vxs[i][1],vxs[j+1][1]], color="b")


link_list = convertMatrix2List(links)

fig = plt.figure(figsize=(4, 4), dpi=80)
ax = fig.add_subplot(111)

t = 0
dt = 0.02
Nt = 40

plot = True
pos = vxs.copy()
vel = np.zeros(pos.shape)
acc = np.zeros(pos.shape)
boxsize = 64 * 16


## Acceleration
# for i in range(Nt):
#     vel += acc * dt/2
#     pos += vel * dt
#     acc = getAcc(pos, link_list)
#     vel += acc * dt/2
#     t+=dt
#     if plot:
#         plt.cla()
#         plt.plot(pos[[link_list[:,0],link_list[:,1]],0], pos[[link_list[:,0],link_list[:,1]],1],color="blue")
#         plt.scatter(pos[:,0],pos[:,1],s=10,color="purple")
#         ax.set(xlim=(0, boxsize), ylim=(0, boxsize))
#         ax.set_aspect('equal','box')
#         plt.savefig(str(i)+'.png', dpi=240)

## Velocity
# for i in range(Nt):
#     pos += vel * dt
#     vel = getVel(pos, link_list)
#     t+=dt
#     if plot:
#         plt.cla()
#         plt.plot(pos[[link_list[:,0],link_list[:,1]],0], pos[[link_list[:,0],link_list[:,1]],1],color="blue")
#         plt.scatter(pos[:,0],pos[:,1],s=10,color="purple")
#         ax.set(xlim=(0, boxsize), ylim=(0, boxsize))
#         ax.set_aspect('equal','box')
#         plt.savefig(str(i)+'.png', dpi=240)


## Velocity with fixed edges

# Computing the alpha shape
edges, b_vxs = alpha_shape(vxs, alpha=200, only_outer=True)
is_out = np.ones([len(vxs), 1])
for i, j in edges:
    is_out[i] = 0
    is_out[j] = 0

k = np.random.randint(len(link_list))
# link_list=link_list[:k,:]+link_list[k+1:,:]
link_list = np.delete(link_list, [k + i for i in range(4)], axis=0)
# print(k)
outer = np.array([vxs[i] for i in range(len(is_out)) if is_out[i] == 0])

left = [
    i for i in range(len(is_out)) if is_out[i] == 0 and vxs[i][0] == np.min(outer, 0)[0]
]
right = [
    i for i in range(len(is_out)) if is_out[i] == 0 and vxs[i][0] == np.max(outer, 0)[0]
]
is_leftright = np.ones([len(vxs), 1])
is_left = np.zeros([len(vxs), 1])
is_right = np.zeros([len(vxs), 1])
for i in left:
    is_leftright[i] = 0
    is_left[i] = 1
for i in right:
    is_leftright[i] = 0
    is_right[i] = 1

c_vel = np.array([1000, 0])


for i in range(Nt):
    pos += vel * is_out * dt
    # pos += vel * is_leftright * dt
    # pos += -c_vel * is_left * dt
    # pos += c_vel * is_right * dt
    vel = getVel(pos, link_list)
    t += dt
    if plot:
        plt.cla()
        plt.plot(
            pos[[link_list[:, 0], link_list[:, 1]], 0],
            pos[[link_list[:, 0], link_list[:, 1]], 1],
            color="blue",
        )
        plt.scatter(pos[:, 0], pos[:, 1], s=10, color="purple")
        # ax.set(xlim=(-500, boxsize + 1400), ylim=(0, boxsize + 600))
        ax.set(xlim=(0, boxsize), ylim=(0, boxsize))
        ax.set_aspect("equal", "box")
        plt.savefig(str(i) + ".png", dpi=240)

ims = []

fig = plt.figure(figsize=(4, 4), dpi=80)
ax = fig.add_subplot(111)

for i in range(Nt):
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    im = ax.imshow(plt.imread(str(i) + ".png"), animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=60, repeat=True)
ani.save("comparison.gif", fps=10)
