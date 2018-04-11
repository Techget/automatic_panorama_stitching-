from __future__ import print_function  #
import cv2
import argparse
import os
import numpy as np
import random
# from sets import Set

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
    detector = cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3,\
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
            y_taken = dict()
            for m,n in nearest_matches:
                if m.distance < 0.7*n.distance and m.distance<GOOD_MATCH_DISTANCE:
                    if (kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1]) in y_taken:
                        if y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])].distance > m.distance:
                            good.remove(y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])])
                            good.append(m)
                            y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])] = m
                    else:
                        good.append(m)
                        y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])] = m

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
    # error = np.sqrt((estimated_p2[0]-p2[0])**2 + (estimated_p2[1]-p2[1])**2)
    # return error
    error = p2 - estimated_p2
    return np.linalg.norm(error)

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
    INLIER_THRESHOLD = 7
    MAX_LOOPS_TO_RUN = 5000
    count = 0
    while(max_inliners_amount < len(corrs)*0.9 and count <= MAX_LOOPS_TO_RUN):
        count += 1
        randomFour = []
        for i in range(0, 4):
            randomFour.append(list(corrs[random.randint(0, len(corrs)-1)]))
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
    detector = cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3,\
        contrastThreshold=0.04,edgeThreshold=10, sigma=1.6)

    GOOD_MATCH_DISTANCE = 100
    GOOD_MATCH_POINTS_AMOUNT = 20
    output_imgs = {}

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
            y_taken = dict()
            for m,n in nearest_matches:
                if m.distance < 0.7*n.distance and m.distance<GOOD_MATCH_DISTANCE:
                    if (kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1]) in y_taken:
                        if y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])].distance > m.distance:
                            good.remove(y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])])
                            good.append(m)
                            y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])] = m
                    else:
                        good.append(m)
                        y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])] = m

            # filter good match pics
            if len(good) < GOOD_MATCH_POINTS_AMOUNT:
                continue
    
            for flag in range(1, 3): 
                correspondence_list = []
                for match in good:
                    (x1, y1) = kp1[match.queryIdx].pt
                    (x2, y2) = kp2[match.trainIdx].pt
                    if flag == 1:
                        correspondence_list.append([x1, y1, x2, y2])
                    else:
                        correspondence_list.append([x2, y2, x1, y1])
                # calcuate h_matrix
                homography_matrix = ransac_homography_matrix(np.matrix(correspondence_list))
                # print(homography_matrix)

                width1, height1, width2, height2 = None, None, None, None
                if flag == 1:
                    width1, height1 = gray1.shape
                    width2, height2 = gray2.shape
                else:
                    width1, height1 = gray2.shape
                    width2, height2 = gray1.shape

                xh = np.linalg.inv(homography_matrix)
                homography_matrix = (1/xh.item(8)) * xh
                indY, indX = np.indices((width1,height1))  # similar to meshgrid/mgrid
                lin_homg_pts = np.stack((indX.ravel(), indY.ravel(), np.ones(indY.size)))
                trans_lin_homg_pts = homography_matrix.dot(lin_homg_pts)
                trans_lin_homg_pts /= trans_lin_homg_pts[2,:]
                map_ind = homography_matrix.dot(lin_homg_pts)
                map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
                map_x = map_x.reshape(width1, height1).astype(np.float32)
                map_y = map_y.reshape(width1, height1).astype(np.float32)

                if flag == 1:
                    dst = cv2.remap(imgs[i], map_x, map_y, cv2.INTER_LINEAR)
                    output_imgs["warped_img_%d(img_%d_reference).jpg"%(i,j)] = dst
                else:
                    dst = cv2.remap(imgs[j], map_x, map_y, cv2.INTER_LINEAR)
                    output_imgs["warped_img_%d(img_%d_reference).jpg"%(j,i)] = dst                  

    return output_imgs


def save_step_3(img_pairs, output_path="./output/step3"):
    """Save the intermediate result from Step 3"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if output_path[:-1]!='/':
        output_path += '/'

    for key, value in img_pairs.iteritems():
        cv2.imwrite("%s%s"%(output_path,key), value)


def up_to_step_4(imgs):
    detector = cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3,\
        contrastThreshold=0.04,edgeThreshold=10, sigma=1.6)

    GOOD_MATCH_DISTANCE = 100
    GOOD_MATCH_POINTS_AMOUNT = 20
    center_image_index = len(imgs)//2

    # left stitch
    img_a = imgs[0]
    for ind in range(1, center_image_index+1):
        gray1= cv2.cvtColor(img_a,cv2.COLOR_BGR2GRAY)
        kp1, des1 = detector.detectAndCompute(gray1,None)
        gray2= cv2.cvtColor(imgs[ind],cv2.COLOR_BGR2GRAY)
        kp2, des2 = detector.detectAndCompute(gray2,None)
        distances = np.sqrt(((des1[:, :, None] - des2[:, :, None].T) ** 2).sum(1))

        nearest_matches = []
        for x in range(0, len(distances)):
            indexes = distances[x].argsort(kind='mergesort')[:2]
            nearest_matches.append((cv2.DMatch(x, indexes[0], distances[x][indexes[0]]), \
                cv2.DMatch(x, indexes[1], distances[x][indexes[1]])))
        good = []
        y_taken = dict()
        for m,n in nearest_matches:
            if m.distance < 0.7*n.distance and m.distance<GOOD_MATCH_DISTANCE:
                if (kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1]) in y_taken:
                    if y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])].distance > m.distance:
                        good.remove(y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])])
                        good.append(m)
                        y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])] = m
                else:
                    good.append(m)
                    y_taken[(kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])] = m

        # filter good match pics
        if len(good) < GOOD_MATCH_POINTS_AMOUNT:
            continue

        correspondence_list = []
        for match in good:
            (x1, y1) = kp1[match.queryIdx].pt
            (x2, y2) = kp2[match.trainIdx].pt
            correspondence_list.append([x1, y1, x2, y2])
        homography_matrix = ransac_homography_matrix(np.matrix(correspondence_list))
        print(homography_matrix)

        result = cv2.warpPerspective(imgs[ind], homography_matrix,\
            (imgs[ind].shape[1] + img_a.shape[1], imgs[ind].shape[0]))
        result[0:img_a.shape[0], 0:img_a.shape[1]] = img_a


        # width1, height1 = gray1.shape
        # width2, height2 = gray2.shape
        # print(width1, width2, height1, height2)
        # xh = np.linalg.inv(homography_matrix)
        # homography_matrix = (1/xh.item(8)) * xh
        # new_image_width = width2
        # new_image_height = height1 + height2
        # indY, indX = np.indices((new_image_width, new_image_height))
        # lin_homg_pts = np.stack((indX.ravel(), indY.ravel(), np.ones(indY.size)))
        # trans_lin_homg_pts = homography_matrix.dot(lin_homg_pts)
        # trans_lin_homg_pts /= trans_lin_homg_pts[2,:]
        # map_ind = homography_matrix.dot(lin_homg_pts)
        # map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
        # map_x = map_x.reshape(new_image_width, new_image_height).astype(np.float32)
        # map_y = map_y.reshape(new_image_width, new_image_height).astype(np.float32)

        # dst = cv2.remap(img_a, map_x, map_y, cv2.INTER_LINEAR)
        # # print(dst.shape)
        # # x = dst[0:width2, height1:height2]
        # # print(x.shape)
        # dst[0:width2, height1:height1+height2] = imgs[ind]
        cv2.imwrite('testq4.jpg', result)




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
    filenames = []
    for filename in sorted(os.listdir(args.input), key=lambda f: int(filter(str.isdigit,f))):
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
        save_step_4(panoramic_img, args.output)
