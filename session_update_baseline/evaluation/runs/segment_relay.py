"""End of a session that is not the last one in a continuous Khronos run: subscribes the player's segment-end topic
(the player waits for a subscriber before it plays) and acknowledges each segment end, so Khronos itself never sees a
finish and keeps its map.   usage: segment_relay.py END_TOPIC ACK_TOPIC"""
import sys
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty

rclpy.init()
node = rclpy.create_node('khronos_continuous_segment_relay')
ack = node.create_publisher(Empty, sys.argv[2], QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
node.create_subscription(Empty, sys.argv[1], lambda _m: ack.publish(Empty()),
                         QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
rclpy.spin(node)
