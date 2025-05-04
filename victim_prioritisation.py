import os
import pandas as pd
import google.generativeai as genai
import math
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Configure Gemini API
genai.configure(api_key='AIzaSyCY6R4TKMk5mZke5N2wKPgH9Ft19ah-6xk')

# Load victim data from Excel
file_path = "/home/boobalan/Downloads/victim_data_coordinates.xlsx"
df = pd.read_excel(file_path)

# Define rescuer's starting location
rescuer_x, rescuer_y = -0.019, 1.09  # Ensure float type

# Define prioritization function
def prioritize_victim(conscious, temperature, emotion, alive, victim_x, victim_y):
    if not alive:
        return -1  # Deceased, no rescue needed
    
    score = 0
    if not conscious:
        score += 3
    if temperature < 35 or temperature > 39:
        score += 2
    if emotion in ["panicked", "distressed"]:
        score += 1
    
    distance = math.sqrt((victim_x - rescuer_x) ** 2 + (victim_y - rescuer_y) ** 2)
    distance_factor = 1 / (distance + 1)
    
    return score * distance_factor

# Apply prioritization function
df["priority_score"] = df.apply(lambda row: prioritize_victim(
    row["conscious"], row["temperature"], row["emotion"], row["alive"], float(row["location_x"]), float(row["location_y"])), axis=1)

# Sort victims by priority
df = df.sort_values(by="priority_score", ascending=False)

# Convert to list of dictionaries
victim_list = df.to_dict(orient="records")

# ROS2 Node for RViz2 Visualization
class RescueVisualization(Node):
    def __init__(self):
        super().__init__('rescue_visualization')
        self.marker_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.publish_markers()

    def publish_markers(self):
        marker_array = MarkerArray()
        marker_id = 0

        # Rescuer marker
        rescuer_marker = Marker()
        rescuer_marker.header.frame_id = "map"
        rescuer_marker.type = Marker.SPHERE
        rescuer_marker.action = Marker.ADD
        rescuer_marker.pose.position.x = float(rescuer_x)
        rescuer_marker.pose.position.y = float(rescuer_y)
        rescuer_marker.pose.position.z = 0.0
        rescuer_marker.scale.x = 0.5
        rescuer_marker.scale.y = 0.5
        rescuer_marker.scale.z = 0.5
        rescuer_marker.color.r = 1.0
        rescuer_marker.color.g = 0.0
        rescuer_marker.color.b = 0.0
        rescuer_marker.color.a = 1.0
        rescuer_marker.id = marker_id
        marker_array.markers.append(rescuer_marker)
        marker_id += 1

        # Victim markers with numbers
        for i, row in enumerate(df.itertuples(), start=1):
            victim_marker = Marker()
            victim_marker.header.frame_id = "map"
            victim_marker.type = Marker.SPHERE
            victim_marker.action = Marker.ADD
            victim_marker.pose.position.x = float(row.location_x)
            victim_marker.pose.position.y = float(row.location_y)
            victim_marker.pose.position.z = 0.0
            victim_marker.scale.x = 0.3
            victim_marker.scale.y = 0.3
            victim_marker.scale.z = 0.3
            victim_marker.color.r = 0.0
            victim_marker.color.g = 0.0
            victim_marker.color.b = 1.0
            victim_marker.color.a = 1.0
            victim_marker.id = marker_id
            marker_array.markers.append(victim_marker)
            marker_id += 1

            # Number label marker
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(row.location_x)
            text_marker.pose.position.y = float(row.location_y)
            text_marker.pose.position.z = 0.5
            text_marker.scale.z = 0.4
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = str(i)
            text_marker.id = marker_id
            marker_array.markers.append(text_marker)
            marker_id += 1

        # Path marker
        path_marker = Marker()
        path_marker.header.frame_id = "map"
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.1
        path_marker.color.r = 0.0
        path_marker.color.g = 1.0
        path_marker.color.b = 0.0
        path_marker.color.a = 1.0
        path_marker.id = marker_id

        # Connect rescuer and victims in order
        path_marker.points.append(Point(x=float(rescuer_x), y=float(rescuer_y), z=0.0))
        for row in df.itertuples():
            path_marker.points.append(Point(x=float(row.location_x), y=float(row.location_y), z=0.0))
        
        marker_array.markers.append(path_marker)
        self.marker_pub.publish(marker_array)

# Start ROS2 Node
def main(args=None):
    # Print prioritization results
    print("Victim Prioritization Results:")
    print(df[["location_x", "location_y", "priority_score"]])

    # Gemini API Output
    prompt = f"Given this list of victims {victim_list}, prioritize them for rescue based on urgency and proximity."
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    print("\nGemini Prioritization Output:")
    print(response.text)

    # Start ROS2 visualization
    rclpy.init(args=args)
    node = RescueVisualization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
