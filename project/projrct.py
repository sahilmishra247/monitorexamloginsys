import cv2

# Load the fingerprint images in grayscale
img1 = cv2.imread(r"E:\python\project\fingerprint2.png", 0)
img2 = cv2.imread(r"E:\python\project\fingerprint1.png", 0)


# Initiate ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (lower is better)
matches = sorted(matches, key=lambda x: x.distance)

print("Number of matches:", len(matches))
if len(matches) > 50:
    print("Likely a Match")
else:
    print("Not a Match")
    
cv2.imshow("output fingerprint1",img1)
cv2.imshow("output fingerprint2",img2)
cv2.waitKey(0)    
