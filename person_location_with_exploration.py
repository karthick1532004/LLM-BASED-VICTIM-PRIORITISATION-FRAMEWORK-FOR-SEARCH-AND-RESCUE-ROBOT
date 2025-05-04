#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool  # Import Bool for controlling exploration
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import defaultdict

class YOLOGoalPublisher(Node):
    def __init__(self):
        super().__init__('yolo_goal_publisher')
        
        self.model = YOLO("yolov8n-pose.pt")
        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Publisher to stop and resume exploration
        self.explore_control_pub = self.create_publisher(Bool, '/explore/resume', 10)
        
        self.depth_image = None
        self.robot_position = (0.0, 0.0)  
        self.robot_orientation = 0.0  
        self.detected_persons = []  
        self.current_goal = None  
        self.detection_history = defaultdict(list)  
        self.detection_threshold = 3  
        
        self.camera_intrinsics = (525.0, 525.0, 320.0, 240.0)  

        self.get_logger().info('YOLO Goal Publisher Node Initialized')

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')
    
    def odom_callback(self, msg):
        self.robot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        _, _, self.robot_orientation = self.euler_from_quaternion(msg.pose.pose.orientation)
        
        # Check if the robot has reached the goal
        if self.current_goal:
            distance_to_goal = math.dist(self.robot_position, self.current_goal[:2])
            if distance_to_goal < 0.5:  
                self.get_logger().info(f'Goal reached at {self.current_goal}, resuming exploration')
                self.current_goal = None
                self.resume_exploration()

    def euler_from_quaternion(self, quat):
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return (0.0, 0.0, yaw)

    def get_world_position(self, bbox, depth):
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        fx, fy, cx_intr, cy_intr = self.camera_intrinsics
        if depth == 0 or fx == 0 or fy == 0:
            return None
        
        X_cam = (cx - cx_intr) * depth / fx
        Y_cam = (cy - cy_intr) * depth / fy
        Z_cam = depth
        
        world_x = self.robot_position[0] + Z_cam * math.cos(self.robot_orientation) + X_cam * math.sin(self.robot_orientation)
        world_y = self.robot_position[1] + Z_cam * math.sin(self.robot_orientation) - X_cam * math.cos(self.robot_orientation)
        
        self.get_logger().info(f'Detected person world position -> X: {world_x}, Y: {world_y}, Z: 0.0')
        return (world_x, world_y, 0.0)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model.predict(cv_image, conf=0.5, classes=[0])
            
            for result in results:
                for i, box in enumerate(result.boxes):
                    bbox = (int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                            int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                    
                    depth = float(self.depth_image[int((bbox[1] + bbox[3]) / 2), int((bbox[0] + bbox[2]) / 2)])
                    person_position = self.get_world_position(bbox, depth)
                    if person_position:
                        self.detection_history[i].append(person_position)
                        if len(self.detection_history[i]) >= self.detection_threshold:
                            avg_position = np.mean(self.detection_history[i][-self.detection_threshold:], axis=0)
                            if tuple(avg_position) not in self.detected_persons:
                                self.detected_persons.append(tuple(avg_position))
                    
                    cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(cv_image, "Person", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if not self.current_goal:
                self.publish_next_goal()
            
            cv2.imshow("YOLO Detection", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def publish_next_goal(self):
        if self.detected_persons:
            self.detected_persons.sort(key=lambda pos: math.dist(self.robot_position, (pos[0], pos[1])))

            candidate_goal = self.detected_persons.pop(0)

            # Ensure we are setting a new goal
            if self.current_goal == candidate_goal:
                return  # Prevent unnecessary goal re-publication

            self.current_goal = candidate_goal

            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = "map"
            goal_msg.pose.position.x = self.current_goal[0]
            goal_msg.pose.position.y = self.current_goal[1]
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.w = 1.0

            self.get_logger().info(f'Published goal pose: {goal_msg.pose.position}')
            
            # **Only stop exploration if a new goal is actually published**
            self.goal_pub.publish(goal_msg)
            if not hasattr(self, 'exploration_stopped') or not self.exploration_stopped:
                self.stop_exploration()
                self.exploration_stopped = True  # Mark that exploration has been stopped



    def stop_exploration(self):
        """Stops exploration by publishing False to /explore/resume"""
        msg = Bool()
        msg.data = False
        self.explore_control_pub.publish(msg)
        self.get_logger().info('Exploration stopped.')

    def resume_exploration(self):
        """Resumes exploration by publishing True to /explore/resume"""
        msg = Bool()
        msg.data = True
        self.explore_control_pub.publish(msg)
        self.get_logger().info('Exploration resumed.')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOGoalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
