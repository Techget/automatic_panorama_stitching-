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

def calculate_homography2(correspondences):
    A = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])
        x, y = p1.item(0), p1.item(1)
        u, v = p2.item(0), p2.item(1)
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H


def ransac_homography_matrix(corrs):
    h_matrix = None
    max_inliners_amount = -1
    LOOPS_TO_RUN = 3000
    INLIER_THRESHOLD = 7

    for i in range(0, LOOPS_TO_RUN):
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

                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                # dst = np.zeros((1212,1366))
                # im_out = cv2.warpPerspective(imgs[i], 0, M, [1212, 1366])
                # cv2.imshow('123',im_out)
                # cv2.waitKey()
                print(homography_matrix)
                print(M)
                print('###')

                # use the h_matrix to create a warped image, with the remap function like process
                # A'=H*A, get the size of newly create image, use A=A'*invH to get the
                # corresponding pixel value in original image
                width1, height1 = None, None
                if flag == 1:
                    width1, height1 = gray1.shape
                else:
                    width1, height1 = gray2.shape

                original_coordinates = [[],[],[]]
                for x in range(0,width1):
                    for y in range(0,height1): 
                        original_coordinates[0].append(x)                   
                        original_coordinates[1].append(y)
                        original_coordinates[2].append(1)
                original_coordinates = np.matrix(np.array(original_coordinates))
                
                transformed_coordinates = homography_matrix.dot(original_coordinates)                
                for x in range(0,len(transformed_coordinates[0])):
                    transformed_coordinates[0][x] /= transformed_coordinates[2][x]
                    transformed_coordinates[1][x] /= transformed_coordinates[2][x]

                # print(original_coordinates)
                # print(transformed_coordinates)
                new_image_width_low = int(np.ceil(min(transformed_coordinates[0])))
                new_image_width_high = int(np.floor(max(transformed_coordinates[0])))
                new_image_height_low = int(np.ceil(min(transformed_coordinates[1])))
                new_image_height_high = int(np.floor(max(transformed_coordinates[1])))

                homography_matrix = np.linalg.inv(homography_matrix)
                # output new image
                print(new_image_width_low, new_image_width_high, new_image_height_low, new_image_height_high)
                output_image = np.zeros((new_image_width_high - new_image_width_low,\
                    new_image_height_high - new_image_height_low, 3))
                for x in range(new_image_width_low, new_image_width_high):
                    for y in range(new_image_height_low, new_image_height_high):
                        p1 = np.matrix([x, y, 1])
                        transformed_coordinate = np.dot(homography_matrix, p1.T)
                        transformed_coordinate = (1/transformed_coordinate.item(2))*transformed_coordinate
                        temp_x,temp_y = transformed_coordinate.item(0),transformed_coordinate.item(1)
                
                        temp_x = int(np.floor(temp_x))
                        temp_y = int(np.floor(temp_y))
                        color = None
                        if temp_x < 0 or temp_x >= width1 or temp_y < 0 or temp_y >= height1:
                            color = [0,0,0]
                        else:
                            color = imgs[i][temp_x][temp_y]

                        if flag == 1:
                            output_image[x-new_image_width_low][y-new_image_height_low] =  color
                        else:
                            output_image[x-new_image_width_low][y-new_image_height_low] = color

                if flag == 1:
                    output_imgs["warped_img_%d(img_%d_reference).jpg"%(i,j)] = output_image
                else:
                    output_imgs["warped_img_%d(img_%d_reference).jpg"%(j,i)] = output_image
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
        save_step_4(panoramic_img, args.output)
