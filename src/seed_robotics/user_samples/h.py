import rospy
import time
while not rospy.is_shutdown():
    for i in range(10):
        print('123456')
        time.sleep(1)
    break