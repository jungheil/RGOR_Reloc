from pypcd import pypcd
import rospy
from sensor_msgs.msg import PointCloud2


pc = pypcd.PointCloud.from_path('map.pcd')
outmsg = pc.to_msg()
pub = rospy.Publisher('outcloud', PointCloud2, cb)

while not rospy.is_shutdown():
    outmsg.header = rospy.Header()
    pub.publish(outmsg)