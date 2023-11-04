import numpy as np
import matplotlib.pyplot as plt

WHEEL_RADIUS=2.5 #cm
AXIS_LENGTH=20 #cm
WHEEL_RATE=10 # RPM
SIZE_OF_GRID=10 #cm

class Robot:
    def __init__(self, x0, y0, theta0, wheel_radius=WHEEL_RADIUS, axis_length=AXIS_LENGTH, wheel_rate=WHEEL_RATE):
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.wheel_radius = wheel_radius
        self.axis_length = axis_length
        # The model we are working is that our engine work as pulse function, 0 or wheel_rate
        self.wheel_rate = wheel_rate

        self.log = [{
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "dist_travelled":0,
            "wheel_rotations":0,
        }]
    
    def __repr__(self):
        return f"Robot(x={self.x}, y={self.y}, theta={self.theta})"

    def move(self, v, w):
        """
        Move the robot with linear velocity v and angular velocity w.
        """
        vr, vl = self.calculate_vr_and_vl_based_on_v_and_w(v, w)
        delta_x, delta_y, delta_theta = self.differential_drive_robot_model(self.theta, vr, vl)
        self.update(delta_x, delta_y, delta_theta)

    def update(self, delta_x, delta_y, delta_theta):
        """
        Update the state of the robot.
        """
        dist_travelled = SIZE_OF_GRID * np.sqrt(delta_x**2 + delta_y**2) # in cm
        if dist_travelled == 0: # rotation
            # considering our robot is a unicycle, we can calculate the wheel rotations
            # based on the angle it turned. We don't care about the direction of the rotation
            # so we will take the absolute value of the angle
            unicycle_wheel_rotations = abs(np.deg2rad(delta_theta) / (2 * np.pi * self.axis_length))
            # because our wheels are synchronous, we turn with double the wheel rate (2w)
            # so we will divide by 2
            wheel_rotations = unicycle_wheel_rotations / 2
        else: # translation
            wheel_rotations = (dist_travelled) / (2 * np.pi * self.wheel_radius)
            

        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta

        self.log.append(
            {
                "x": self.x,
                "y": self.y,
                "theta": self.theta,
                "dist_travelled":dist_travelled,
                "wheel_rotations":wheel_rotations,
            }
        )

    def move_to_pose(self, x, y, theta):
        """
        Move the robot to a point with a certain angle (x,y,theta) in the world.
        """
        adjust_angle_to_move = np.arctan2(y-self.y, x-self.x)
        self.move(0, adjust_angle_to_move-self.theta)
        # move in a straight line
        self.move(np.sqrt((x-self.x)**2 + (y-self.y)**2), 0)
        # adjust angle to theta
        self.move(0, theta-self.theta)

    def differential_drive_robot_model(self, theta0, vr, vl) -> np.ndarray:
        """
        Differential drive robot model.

        Args:
            x0: x coordinate of the robot
            y0: y coordinate of the robot
            theta0: orientation of the robot
            vr: right wheel velocity
            vl: left wheel velocity
            
        Returns:
            np.ndarray: new state of the robot, with shape (3,)
                representing (x, y, theta)
        """
        delta_theta = (self.wheel_radius * (vr - vl) / self.axis_length)
        delta_x = (self.wheel_radius * (vr + vl) / 2 * np.cos(theta0))
        delta_y = self.wheel_radius * (vr + vl) / 2 * np.sin(theta0)

        return np.array([delta_x,delta_y,delta_theta])

    def calculate_vr_and_vl_based_on_v_and_w(self, v, w):
        """
        Calculate vr and vl based on v and w.

        Args:
            v: linear velocity
            w: angular velocity
        
        Returns:
            tuple: (vr, vl)
        """
        vr = (2 * v + w * AXIS_LENGTH) / (2 * self.wheel_radius)
        vl = (2 * v - w * AXIS_LENGTH) / (2 * self.wheel_radius)
        return vr, vl


def draw_trajectory(robot: Robot):
    # Desenhe a rota do robô
    plt.figure(figsize=(10, 8))
    x_values = [log["x"] for log in robot.log]
    y_values = [log["y"] for log in robot.log]
    plt.plot(x_values, y_values, marker='x', linestyle='-')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Robot Trajectory')
    plt.grid(True)
    plt.show()


def main(initial_state, pose_list):
    print("Robot Simulator")

    print("Pose list")
    print(pose_list)

    robot = Robot(*initial_state)
    print("Initial state")
    print(robot)
    
    for x,y,theta in pose_list:
        robot.move_to_pose(x,y,theta)
    
    print("Final state")
    print(robot)

    print("Logs")
    for log in robot.log:
        print(log)
    print()
    print("Total distance travelled (cm): ", sum([log["dist_travelled"] for log in robot.log]))
    print("Total wheel rotations: ", sum([log["wheel_rotations"] for log in robot.log]))

    draw_trajectory(robot)

if __name__ == "__main__":
    initial_state = (0, 0, 0)
    pose_ordered = [(4,6,45),(6,8,90)]
    main(initial_state, pose_ordered)