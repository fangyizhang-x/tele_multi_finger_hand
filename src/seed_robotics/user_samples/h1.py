import rospy
import time
from std_msgs.msg import Int32
if __name__ == '__main__':
    rospy.init_node('outputnumber')
    rate = rospy.Rate(5)
    pub = rospy.Publisher('/talker', Int32, queue_size=10)
    rospy.loginfo("Publisher has been started.")
    while not rospy.is_shutdown():
        pub.publish(2023)
        rospy.loginfo(2023)
        rate.sleep()
        while not rospy.is_shutdown():
            pub.publish(2024)
            rospy.loginfo(2024)
            rate.sleep()
