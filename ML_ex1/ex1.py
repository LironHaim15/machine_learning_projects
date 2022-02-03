# Liron Haim 206234635
import matplotlib.pyplot as plt
import numpy as np
import sys


def average(array):
    # given an array of pixels, this function will return the average pixel.
    if len(array) == 0:
        return np.array([-1, -1, -1])
    return sum(array) / len(array)


def get_loss(ps, cents):
    # given a centroid and an array of pixels, this function will return the average distance between
    # the pixels and centroids.
    sum_dis = 0

    for p in ps:
        min_dis = np.sum(pow(p - cents[0][0], 2))  # set minimal distance as the distance from the first
        dis = min_dis
        # check minimal distance
        for c in cents:
            dis = np.sum(pow(p - c[0], 2))
            if min_dis > dis:
                min_dis = dis
        sum_dis = sum_dis + min_dis  # sum distances

    return sum_dis / len(ps)


# save arguments ,read image and centroids text file.
image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
initial_centroids = np.loadtxt(centroids_fname)
orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float) / 255. # normalize pixels' value
pixels = pixels.reshape(-1, 3) # reshape pixels array

centroids = []
stop = False # stopping flag. True when all the centroids become fixed.

# initiate an empty list for each centroid.
for centroid in initial_centroids:
    centroids.append([centroid, []])

output_file = open(out_fname, "w")  # open or create output file

for iter in range(20):  # max 20 iteration or until the algorithm converges
    if stop:
        break

    # reset sorted pixels' list for each centroid.
    for centroid in centroids:
        centroid[1] = []

    for pixel in pixels:
        closest_cent = centroids[0]  # first centroid in the centroids array
        min_distance = np.sum(pow(pixel - closest_cent[0], 2))  # set minimal distance as the distance from the first

        # check minimal distance
        for centroid in centroids:
            distance = np.sum(pow(pixel - centroid[0], 2))
            if min_distance > distance:
                min_distance = distance
                closest_cent = centroid
        closest_cent[1].append(pixel)  # sort the pixel into suitable centroid

    stop = True
    for centroid in centroids:
        avg = average(centroid[1]).round(4)  # calculates average pixel
        if np.array_equal(avg, np.array([-1, -1, -1])):
            continue
        if not np.array_equal(avg, centroid[0]):
            centroid[0] = avg   # set the average pixel as the new centroid.
            stop = False

    print(get_loss(pixels, centroids))
    # write to the output file the new centroids and iteration number.
    output_file.write(f"[iter {iter}]:{','.join([str(centroid[0]) for centroid in centroids])}\n")

output_file.close()  # close file
