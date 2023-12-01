import signal
import sys
from enum import Enum

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .turtle_laser import LaserSub

SPEED = 0.1
ROTATION_SPEED = 1.0


class Turn(Enum):
    LEFT = ROTATION_SPEED
    RIGHT = -ROTATION_SPEED * 0.5


class State(Enum):
    FORWARD = 0
    LEFT = 2
    RIGHT = 3
    FAST_FORWARD = 5
    STOPPED = 10


class CustomTurtle(Node):
    def __init__(self):
        super().__init__("velocidadepub")
        self.velocity_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.laser = LaserSub()
        self.state = State.FORWARD
        self.next_state = State.FORWARD

    def main_loop(self):
        while True:
            print(f"state: {self.state}")
            rclpy.spin_once(self.laser)

            if self.state == State.LEFT:
                self.next_state = State.FORWARD
            else:
                self.set_next_state()

            self.state, self.next_state = self.next_state, State.STOPPED

            if self.state == State.STOPPED:
                self.stop()
            elif self.state == State.FORWARD:
                self.move_forward(SPEED)
            elif self.state == State.FAST_FORWARD:
                self.move_forward(SPEED * 1)
            elif self.state == State.LEFT:
                self.rotate(Turn.LEFT)
            elif self.state == State.RIGHT:
                self.rotate(Turn.RIGHT)

    def set_next_state(self):
        rclpy.spin_once(self.laser)

        if self.laser.frente < 0.25 or self.laser.diagDir < 0.25:
            self.next_state = State.RIGHT
        elif self.laser.esquerda > 0.2:
            self.next_state = State.LEFT
        elif self.state in [State.FORWARD, State.FAST_FORWARD]:
            self.next_state = State.FAST_FORWARD
        else:
            self.next_state = State.FORWARD

    def move_forward(self, speed):
        move_cmd = Twist()
        move_cmd.linear.x = speed
        move_cmd.angular.z = 0.0

        self.velocity_publisher.publish(move_cmd)

    def rotate(self, rotation_speed: Turn):
        move_cmd = Twist()
        move_cmd.linear.x = 0.0
        move_cmd.angular.z = rotation_speed.value

        self.velocity_publisher.publish(move_cmd)

    def stop(self):
        move_cmd = Twist()
        move_cmd.linear.x = 0.0
        move_cmd.angular.z = 0.0

        self.velocity_publisher.publish(move_cmd)


def main(args=None):
    rclpy.init(args=args)

    michelangelo = CustomTurtle()
    michelangelo.main_loop()


if __name__ == "__main__":
    main()
