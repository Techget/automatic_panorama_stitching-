from __future__ import print_function  #
import cv2
import argparse
import os
import numpy as np
import random

def up_to_step_1(imgs):
    """Complete pipeline up to step 3: Detecting features and descriptors"""
    detector = cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3,\
        contrastThreshold=0.04,edgeThreshold=10, sigma=1.6)

    output_imgs = []
    for img in imgs:
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp = detector.detect(gray,None)
        output_img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        output_imgs.append(output_img)

    return output_imgs


def save_step_1(imgs, output_path='./output/step1'):
    """Save the intermediate result from Step 1"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if output_path[:-1]!='/':
        output_path += '/'

    i = 0
    for img in imgs:
        cv2.imwrite("%soutput%d.jpg"%(output_path,i), img)
        i += 1


def up_to_step_2(imgs):
    """Complete pipeline up to step 2: Calculate matching feature points"""
    detector = cv2.xfeatures2d.SIFT_create(nfeatures=300,nOctaveLayers=3,\
        contrastThreshold=0.04,edgeThreshold=10, sigma=1.6)

    GOOD_MATCH_DISTANCE = 100
    GOOD_MATCH_POINTS_AMOUNT = 20

    output_match_images = []
    match_list = []

    for i in range(0,len(imgs)-1):
        for j in range(i+1,len(imgs)):
            gray1= cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY)
            kp1, des1 = detector.detectAndCompute(gray1,None)
            gray2= cv2.cvtColor(imgs[j],cv2.COLOR_BGR2GRAY)
            kp2, des2 = detector.detectAndCompute(gray2,None)
            distances = np.sqrt(((des1[:, :, None] - des2[:, :, None].T) ** 2).sum(1))

            nearest_matches = []
            for x in range(0, len(distances)):
                indexes = distances[x].argsort(kind='mergesort')[:2]
                nearest_matches.append((cv2.DMatch(x, indexes[0], distances[x][indexes[0]]), \
                    cv2.DMatch(x, indexes[1], distances[x][indexes[1]])))

            good = []
            for m,n in nearest_matches:
                if m.distance < 0.75*n.distance and m.distance<GOOD_MATCH_DISTANCE:
                    good.append(m)

            # filter good match pics
            if len(good) > GOOD_MATCH_POINTS_AMOUNT:
                output_match_images.append(cv2.drawMatches(gray1,kp1,gray2,kp2,good, None,flags=2))
                match_list.append("imgA%d_%d_imgB%d_%d_%d.jpg"%(i,len(kp1),j,len(kp2),len(good)))

    return output_match_images, match_list


def save_step_2(match_images, match_list, output_path="./output/step2"):
    """Save the intermediate result from Step 2"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if output_path[:-1]!='/':
        output_path += '/'

    for i in range(0, len(match_images)):
        cv2.imwrite("%s%s"%(output_path,match_list[i]), match_images[i])

def calculate_h_squared_distance(coordinates_list, h):
    coordinates_list = list(np.array(coordinates_list))[0]
    p1 = np.matrix([coordinates_list[0], coordinates_list[1], 1])
    estimated_p2 = np.dot(h, p1.T)
    # normalize the estimated p2 acoording to the 
    # scale parameter(the third paramter in coordinates)
    estimated_p2 = (1/estimated_p2.item(2))*estimated_p2

    p2 = np.transpose(np.matrix([coordinates_list[2], coordinates_list[3], 1]))
    error = np.sqrt((estimated_p2[0]-p2[0])**2 + (estimated_p2[1]-p2[1])**2)
    return error

def calculate_homography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h

def ransac_homography_matrix(corrs):
    h_matrix = None
    max_inliners_amount = -1
    LOOPS_TO_RUN = 1000
    INLIER_THRESHOLD = 5

    for i in range(0, LOOPS_TO_RUN):
        randomFour = []
        for i in range(0, 4):
            randomFour.append(list(corrs[random.randint(0, len(corrs))]))
        randomFour = np.array(randomFour)

        # calculate homography
        h = calculate_homography(randomFour)
        # evaluate homography matrix
        inliers_amount = 0
        for i in range(len(corrs)):
            if calculate_h_squared_distance(corrs[i], h) < INLIER_THRESHOLD:
                inliers_amount += 1

        if inliers_amount > max_inliners_amount:
            max_inliners_amount = inliers_amount
            h_matrix = h

    return h_matrix

def up_to_step_3(imgs):
    """Complete pipeline up to step 3: estimating homographies and warpings"""
    detector = cv2.xfeatures2d.SIFT_create(nfeatures=300,nOctaveLayers=3,\
        contrastThreshold=0.04,edgeThreshold=10, sigma=1.6)

    GOOD_MATCH_DISTANCE = 100
    GOOD_MATCH_POINTS_AMOUNT = 20

    good_matches = []
    matched_img_name_list = []
    matched_descriptors = []
    matched_key_points = []

    for i in range(0,len(imgs)-1):
        for j in range(i+1,len(imgs)):
            gray1= cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY)
            kp1, des1 = detector.detectAndCompute(gray1,None)
            gray2= cv2.cvtColor(imgs[j],cv2.COLOR_BGR2GRAY)
            kp2, des2 = detector.detectAndCompute(gray2,None)
            distances = np.sqrt(((des1[:, :, None] - des2[:, :, None].T) ** 2).sum(1))

            nearest_matches = []
            for x in range(0, len(distances)):
                indexes = distances[x].argsort(kind='mergesort')[:2]
                nearest_matches.append((cv2.DMatch(x, indexes[0], distances[x][indexes[0]]), \
                    cv2.DMatch(x, indexes[1], distances[x][indexes[1]])))

            good = []
            for m,n in nearest_matches:
                if m.distance < 0.75*n.distance and m.distance<GOOD_MATCH_DISTANCE:
                    good.append(m)

            # filter good match pics
            if len(good) > GOOD_MATCH_POINTS_AMOUNT:
                good_matches.append(good)
                matched_img_name_list.append((i,j))
                matched_descriptors.append((des1, des2))
                matched_key_points.append((kp1, kp2))
    
    output_imgs = []
    for i in range(0, len(good_matches)):
        correspondence_list = []
        for match in good_matches[i]:
            (x1, y1) = matched_key_points[i][0][match.queryIdx].pt
            (x2, y2) = matched_key_points[i][1][match.trainIdx].pt
            correspondence_list.append([x1, y1, x2, y2])
        corrs = np.matrix(correspondence_list)
        # calcuate h_matrix
        homography_matrix = ransac_homography_matrix(corrs)
        # use the h_matrix to create a warped image, with the remap function
        # A'=H*A, get the size of newly create image, use A=A'*H to get the
        # corresponding pixel value in original image
        print(homography_matrix)




def save_step_3(img_pairs, output_path="./output/step3"):
    """Save the intermediate result from Step 3"""
    # ... your code here ...
    pass


def up_to_step_4(imgs):
    """Complete the pipeline and generate a panoramic image"""
    # ... your code here ...
    return imgs[0]


def save_step_4(imgs, output_path="./output/step4"):
    """Save the intermediate result from Step 4"""
    # ... your code here ...
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        help="compute image stitching pipeline up to this step",
        type=int
    )

    parser.add_argument(
        "input",
        help="a folder to read in the input images",
        type=str
    )

    parser.add_argument(
        "output",
        help="a folder to save the outputs",
        type=str
    )

    args = parser.parse_args()

    imgs = []
    imgs_names = []
    for filename in os.listdir(args.input):
        print(filename)
        img = cv2.imread(os.path.join(args.input, filename))
        imgs.append(img)

    if args.step == 1:
        print("Running step 1")
        modified_imgs = up_to_step_1(imgs)
        save_step_1(modified_imgs, args.output)
    elif args.step == 2:
        print("Running step 2")
        modified_imgs, match_list = up_to_step_2(imgs)
        save_step_2(modified_imgs, match_list, args.output)
    elif args.step == 3:
        print("Running step 3")
        img_pairs = up_to_step_3(imgs)
        save_step_3(img_pairs, args.output)
    elif args.step == 4:
        print("Running step 4")
        panoramic_img = up_to_step_4(imgs)
        save_step_4(img_pairs, args.output)
