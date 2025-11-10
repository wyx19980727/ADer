import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

# img1 = cv.imread('/home/albus/DataSets/REAL-IAD/realiad_256/audiojack/OK/S0001/audiojack_0001_OK_C2_20231021130235.jpg',0)
# img2 = cv.imread('/home/albus/DataSets/REAL-IAD/realiad_256/audiojack/OK/S0001/audiojack_0001_OK_C3_20231021130235.jpg',0)
img1 = cv.imread('/home/albus/DataSets/REAL-IAD/realiad_256/sim_card_set/OK/S0001/sim_card_set_0001_OK_C2_20230922140928.jpg',0)
img2 = cv.imread('/home/albus/DataSets/REAL-IAD/realiad_256/sim_card_set/OK/S0001/sim_card_set_0001_OK_C3_20230922140928.jpg',0)
# img1 = cv.resize(img1, (640,480))
# img2 = cv.resize(img2, (640,480))

sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good = []
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1, pts2, cv.USAC_MAGSAC)

# 640, 480, CV2_USAC_MAGSAC
#F = np.array([[-1.7952282563118947e-05, -2.5656136717715985e-06, -0.003663562133424791], [-1.5624891732428684e-06, -1.984849641888604e-05, 0.007997922515550374], [-0.0026768717699973865, -0.0039014403545582074, 0.9449592234862138]])
# 256, 256, CV2_USAC_MAGSAC
#F = np.array([[-1.1029684979963044e-05, -2.7043940316742407e-06, -0.005064047704338784], [-2.285218389037592e-06, -1.2154109789649656e-05, 0.008547652275275004], [-0.004639108389131278, -0.0052507742945871235, 1.2748714084138606]])
# 256, 256, FM_RANSAC NO PARAMETER
#F = np.array([[1.993421074079428e-05, -4.6161827098430025e-07, -0.00446411389704231], [-2.0091951192069729e-07, 2.0731518550900304e-05, -0.002976929228372168], [-0.0017463410556247897, -0.004136041679059278, 1.0]])
# 256, 256, CV2_USAC_MAGSAC NO PARAMETER
#F = np.array([[1.2601241022814054e-05, 2.576540158417931e-06, 0.004486272497906089], [1.8741274070672324e-06, 1.4007203192777504e-05, -0.007784480305264675], [0.00365114523043757, 0.004668611235231173, -1.133197736355526]])


print(F)

# import ipdb; ipdb.set_trace()

# F = np.array([[]], dtype=np.float64)
#F = np.array([[3.0143358470511253e-05, 0.0002085771575980403, -0.013193311810854619], [-0.00015576795369918038, -1.1512254763493266e-05, 0.03169319947069912], [0.0015459885391127368, -0.04580228869887094, 1.2564756550572522]], dtype=np.float64)
#F = np.array([[0.000010245412938359336, 0.00010199683321295412, -0.011925602837355564], [-0.00012084429010658868, 0.00003767085229920515, 0.031211513170550098], [0.008910858471804544, -0.03547343048480462, 0.5827725608848202]], dtype=np.float64)

#F = np.array([[-2.9470707929872554e-05, 4.497253462022785e-05, 0.009321176180277824], [-7.344026805882998e-05, 2.26290259531242e-05, 0.02375064904293248], [0.001767815030926779, -0.023707022780805064, -0.4364002920233576]], dtype=np.float64)
# F = np.array([[-2.4554061707712235e-05, 0.00016435427363360952, -0.003734527450624095], [-0.0002196998458254249, 9.443372433169353e-06, 0.04782378964190472], [0.013884213702039608, -0.03218369942574307, -1.2660561070995056]])

# print(F)
# We select only inlier points
# import ipdb; ipdb.set_trace()
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
# plt.show()
plt.imsave('./albus/epipolar/epipolar_match_1.png',img5)
plt.imsave('./albus/epipolar/epipolar_match_2.png',img6)
