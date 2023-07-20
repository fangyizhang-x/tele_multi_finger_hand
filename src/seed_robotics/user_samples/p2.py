import rospy
from std_msgs.msg import String
import time

rospy.init_node("counter_publisher")
rate = rospy.Rate(5)
pub = rospy.Publisher("counter",String, queue_size=10)
rospy.loginfo("Publisher has been started.")

while not rospy.is_shutdown():
    msg = String()
    msg.data = 'hahahah'
    pub.publish(msg)
    rospy.loginfo('test11111')
    print('kkk',msg)
    rate.sleep()