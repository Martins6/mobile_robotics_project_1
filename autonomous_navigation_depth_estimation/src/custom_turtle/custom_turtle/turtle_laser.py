import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LaserSub(Node):
    def __init__(self):
        super().__init__('lasersub')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.listener_callback,
            rclpy.qos.qos_profile_sensor_data
        )

    def listener_callback(self, msg):
        self.frente = msg.ranges[0]
        self.diagDir = msg.ranges[45]
        self.esquerda = msg.ranges[90]
        self.tras = msg.ranges[180]
        self.direita = msg.ranges[270]
        self.diagEsq = msg.ranges[315]

def main(args=None):
    rclpy.init(args=args)
    laser_sub = LaserSub()
    rclpy.spin(laser_sub)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
