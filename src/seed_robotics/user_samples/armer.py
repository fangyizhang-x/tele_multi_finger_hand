import rospy
import time
import numpy
import actionlib
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped

from control_msgs.msg import GripperCommandAction, GripperCommandActionGoal
from franka_gripper.msg import GraspAction, GraspActionGoal
from franka_gripper.msg import HomingAction, HomingActionGoal
from franka_gripper.msg import StopAction, StopActionGoal
from franka_gripper.msg import MoveAction, MoveActionGoal

from armer_msgs.msg import MoveToNamedPoseAction, MoveToNamedPoseGoal
from armer_msgs.msg import MoveToPoseAction, MoveToPoseGoal
from armer_msgs.msg import JointVelocity

# rospy.init_node('gentle_benchmark')

class GentleBenchmarkController(object):
    def __init__(self):
        self.setup()
        
    def test(self,x,y,z):
        ori =[]
        self.target = PoseStamped()
        self.target.pose.position.x = x
        self.target.pose.position.y = y
        self.target.pose.position.z = z
        self.target.pose.orientation.x = ori[0]
        self.target.pose.orientation.y =  ori[1]
        self.target.pose.orientation.z =  ori[2]
        self.target.pose.orientation.w =  ori[3]


        goal = MoveToPoseGoal()
        goal.pose_stamped = self.target
        self.move_client.send_goal(goal)
        self.move_client.wait_for_result()
    
    
    def shakeHandDemo(self):
        
        print("--Moving to shaking_hand1")
        self.move_to_somewhere('shaking_hand1')
        print("--Closing gripper")
        self.close_gripper(1)

        print('--move to shaking_up position')
        self.move_to_somewhere('shaking_up')

        print("--Moving to shaking_hand2")
        self.move_to_somewhere('shaking_hand2')

        print("--Openning gripper")
        self.open_gripper()

        print("--Moving to thor_home")
        self.move_to_somewhere('thor_home')

    def run_trial(self):
        # self.do_shake()

        # print("Reseting gripper")
        # self.reset_gripper()

        # print("Openning gripper")
        # self.open_gripper()

        print("Doing shake")
        # self.do_shake()

        print("Moving to drop")
        self.move_to_drop()

        print("Openning gripper")
        self.open_gripper()

    def do_shake(self):
        # self.do_shake_named_pose()
        self.do_shake_joint_vel()
        # self.do_shake_vel()

    def do_shake_joint_vel(self):
        joint_vel = JointVelocity()
        joint_vel.joints = [0]*7

        SHAKE_PERIOD = 0.5
        N_SHAKES = 10
        SHAKE_MAGNITUDE = 0.5
        start_shake_time = time.time()
        while 1:
            current_time = time.time() - start_shake_time
            if current_time > SHAKE_PERIOD * N_SHAKES:
                break
            vel = -SHAKE_MAGNITUDE * numpy.sin(2 * numpy.pi * current_time / SHAKE_PERIOD)
            joint_vel.joints[1] = vel
            joint_vel.joints[6] = vel
            self.joint_vel_publisher.publish(joint_vel)
            rospy.sleep(1./1000)
        
        joint_vel.joints = [0]*7
        self.joint_vel_publisher.publish(joint_vel)
        rospy.sleep(1.)

    def do_shake_named_pose(self):
        for i in range(10):
            self.move_to_shake_1()
            self.move_to_shake_2()
        rospy.sleep(1)

    def do_shake_pose(self):
        target = PoseStamped()
        target.header.frame_id = 'panda_link0'

        target.pose.orientation.x = -1
        target.pose.orientation.y =  0
        target.pose.orientation.z =  0
        target.pose.orientation.w =  0

        target.pose.position.x = 0.6
        target.pose.position.y = 0.0

        speed = 0.5

        for i in range(5):
           
            target.pose.position.z = 0.3
            self.move_to_pose(target, speed)

            target.pose.position.z = 0.2
            self.move_to_pose(target, speed)

    def do_shake_vel(self):
        velocity = TwistStamped()
        velocity.header.frame_id = "panda_EE"
        SHAKE_PERIOD = 0.5
        N_SHAKES = 5
        SHAKE_MAGNITUDE = 0.2
        start_shake_time = time.time()
        while 1:
            current_time = time.time() - start_shake_time
            if current_time > SHAKE_PERIOD * N_SHAKES:
                break
            velocity.twist.linear.z = -SHAKE_MAGNITUDE * numpy.sin(2 * numpy.pi * current_time / SHAKE_PERIOD)
            self.vel_publisher.publish(velocity)
            rospy.sleep(1./1000)

        self.vel_publisher.publish(TwistStamped())
        rospy.sleep(1)

    def setup(self):
        print("Waiting for named pose client")
        self.named_pose_client = actionlib.SimpleActionClient('/arm/joint/named', MoveToNamedPoseAction)
        self.named_pose_client.wait_for_server()

        print("Waiting for gripper client")
        self.gripper_cmd_action = actionlib.SimpleActionClient('/franka_gripper/gripper_action', GripperCommandAction)
        self.gripper_cmd_action.wait_for_server()
        self.gripper_stop_action = actionlib.SimpleActionClient('/franka_gripper/stop', StopAction)
        self.gripper_stop_action.wait_for_server()
        self.gripper_homing_action = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
        self.gripper_homing_action.wait_for_server()
        self.gripper_grasp_action = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.gripper_grasp_action.wait_for_server()
        self.gripper_move_action = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
        self.gripper_move_action.wait_for_server()

        print("Waiting for cartesian client")
        self.move_client = actionlib.SimpleActionClient('/arm/cartesian/pose', MoveToPoseAction)
        self.move_client.wait_for_server()

        print("Starting velocity publisher")
        self.vel_publisher = rospy.Publisher('/arm/cartesian/velocity', TwistStamped, queue_size=1)

        print("Starting joint velocity publisher")
        self.joint_vel_publisher = rospy.Publisher('/arm/joint/velocity', JointVelocity, queue_size=1)

        print("Finished setup")

    def move_to_pose(self, pose, speed=0.15):
        goal = MoveToPoseGoal(goal_pose=pose)
        goal.speed = speed
        self.move_client.send_goal(goal)
        self.move_client.wait_for_result()


    def reset_gripper(self):
        stop_goal = StopActionGoal()
        self.gripper_stop_action.send_goal(stop_goal.goal)
        self.gripper_stop_action.wait_for_result()
        homing_goal = HomingActionGoal()
        self.gripper_homing_action.send_goal(homing_goal.goal)
        self.gripper_homing_action.wait_for_result()


    def open_gripper(self):
        stop_goal = StopActionGoal()
        self.gripper_stop_action.send_goal(stop_goal.goal)
        self.gripper_stop_action.wait_for_result()
        move_goal = MoveActionGoal()
        move_goal.goal.width = 0.08
        move_goal.goal.speed = 0.1
        self.gripper_move_action.send_goal(move_goal.goal)
        self.gripper_move_action.wait_for_result()

    def close_gripper(self, force=0.01):
        grasp_goal = GraspActionGoal()
        # gripper_goal.goal.mode = 1
        # gripper_goal.goal.width = 0.0
        grasp_goal.goal.width = 0.03
        grasp_goal.goal.epsilon.inner = 0.03
        grasp_goal.goal.epsilon.outer = 0.05
        grasp_goal.goal.speed = 0.01
        grasp_goal.goal.force = force
        self.gripper_grasp_action.send_goal(grasp_goal.goal)
        self.gripper_grasp_action.wait_for_result()
    
    def close_gripper0(self):
        gripper_goal = GripperCommandActionGoal()
        gripper_goal.goal.command.position = 0.01
        gripper_goal.goal.command.max_effort = 0.01
        self.gripper_cmd_action.send_goal(gripper_goal.goal)
        self.gripper_cmd_action.wait_for_result()

    def move_to_home(self):
        goal = MoveToNamedPoseGoal()
        goal.pose_name = "gentleMAN_home"
        self.named_pose_client.send_goal(goal)
        self.named_pose_client.wait_for_result()
    
    def move_to_hold(self):
        goal = MoveToNamedPoseGoal()
        goal.pose_name = "gentleMAN_hold"
        self.named_pose_client.send_goal(goal)
        self.named_pose_client.wait_for_result()

    def move_to_shake_1(self):
        goal = MoveToNamedPoseGoal()
        goal.pose_name = "gentleMAN_shake_1"
        self.named_pose_client.send_goal(goal)
        self.named_pose_client.wait_for_result()
    
    def move_to_shake_2(self):
        goal = MoveToNamedPoseGoal()
        goal.pose_name = "gentleMAN_shake_2"
        self.named_pose_client.send_goal(goal)
        self.named_pose_client.wait_for_result()

    def move_to_drop1(self):
        goal = MoveToNamedPoseGoal()
        goal.pose_name = "shaking_hand1"
        # goal.pose_name = "gentleMAN_drop"
        self.named_pose_client.send_goal(goal)
        self.named_pose_client.wait_for_result()

    def move_to_drop2(self):
        goal = MoveToNamedPoseGoal()
        goal.pose_name = "shaking_hand2"
        # goal.pose_name = "gentleMAN_drop"
        self.named_pose_client.send_goal(goal)
        self.named_pose_client.wait_for_result()

    def move_to_grip(self):
        goal = MoveToNamedPoseGoal()
        goal.pose_name = "gentleMAN_grasp"
        self.named_pose_client.send_goal(goal)
        self.named_pose_client.wait_for_result()
    
    def move_to_somewhere(self, pos):
        goal = MoveToNamedPoseGoal()
        goal.pose_name = pos
        # goal.pose_name = "gentleMAN_drop"
        self.named_pose_client.send_goal(goal)
        self.named_pose_client.wait_for_result()

if __name__ == "__main__":
    gbc = GentleBenchmarkController()
    gbc.shakeHandDemo()