import numpy as np
import genesis as gs
import time
import threading
from typing import List, Dict, Any

class GenesisSimulator:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GenesisSimulator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not GenesisSimulator._initialized:
            try:
                gs.init(backend=gs.metal)
            except gs.GenesisException:
                pass
            self.dt = 0.01
            self.scene = None
            self.robot = None
            GenesisSimulator._initialized = True
        
    def create_scene(self):
        if self.scene is not None:
            self.scene = None
            self.robot = None
            
        self.scene = gs.Scene(
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = (0, -3.5, 2.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 30,
                max_FPS       = 60,
            ),
            sim_options = gs.options.SimOptions(
                dt = self.dt,
            ),
            show_viewer = True,
        )
        # add ground plane
        self.scene.add_entity(
            gs.morphs.Plane(),
        )
        # add Franka robot
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file  = 'xml/franka_emika_panda/panda_simple.xml',
                pos   = (0.0, 0.0, 0.0),
                euler = (0, 0, 0),
            ),
        )
        self.scene.build()
        
        # number of degrees of freedom from the robot's state
        self.num_joints = len(self.robot.get_dofs_position())
        print(f"Robot has {self.num_joints} degrees of freedom")
        self.ee_link_idx = len(self.robot.links) - 1
        print(f"End effector link index: {self.ee_link_idx}")
        print(f"Link positions: {self.robot.get_links_pos()}")

    def get_end_effector_position(self):
        pos = self.robot.get_links_pos()[self.ee_link_idx]
        return pos.cpu().numpy()

    def get_joint_positions(self):
        pos = self.robot.get_dofs_position()
        return pos.cpu().numpy()
    
    def get_joint_velocities(self):
        vel = self.robot.get_dofs_velocity()
        return vel.cpu().numpy()
    
    def set_joint_velocities(self, velocities):
        if len(velocities) != self.num_joints:
            velocities = np.zeros(self.num_joints, dtype=np.float32)
        velocities = gs.tensor(velocities.astype(np.float32))
        self.robot.set_dofs_velocity(velocities)

    def setup_control_system_demo(self, system_type: str):
        if not self.scene:
            self.create_scene()
            
        if system_type == "position_control":
            joint_positions = gs.tensor(np.zeros(self.num_joints, dtype=np.float32))
            self.robot.set_dofs_position(joint_positions)
            return {
                'target': [1.0, 0.0, 0.5],
                'description': """
                Position Control Demo:
                - Robot starts at home position
                - Target position at (1, 0, 0.5)
                - Observe how PID parameters affect:
                  * Settling time
                  * Overshoot
                  * Steady-state error
                """
            }
        
        elif system_type == "trajectory_tracking":
            joint_positions = gs.tensor(np.zeros(self.num_joints, dtype=np.float32))
            self.robot.set_dofs_position(joint_positions)
            return {
                'trajectory': lambda t: [np.sin(0.5 * t), np.cos(0.5 * t), 0.5],
                'description': """
                Trajectory Tracking Demo:
                - Robot follows a circular trajectory
                - Demonstrates continuous control
                - Observe tracking error and control effort
                """
            }
            
        elif system_type == "disturbance_rejection":
            joint_positions = np.zeros(self.num_joints, dtype=np.float32)
            joint_positions[0] = 0.5  # Set first joint to 0.5 radians
            joint_positions = gs.tensor(joint_positions)
            self.robot.set_dofs_position(joint_positions)
            return {
                'target': [0.5, 0, 0.5],
                'disturbance': lambda t: [0.2 * np.sin(t), 0, 0],
                'description': """
                Disturbance Rejection Demo:
                - Robot maintains position under external forces
                - Periodic disturbance applied
                - Observe how control system rejects disturbances
                """
            }

    def apply_control(self, current_state: Dict[str, Any], target_state: Dict[str, Any], 
                     controller_params: Dict[str, float]) -> np.ndarray:
        kp = float(controller_params.get('kp', 1.0))
        ki = float(controller_params.get('ki', 0.1))
        kd = float(controller_params.get('kd', 0.05))
        
        # calculate error in task space
        current_pos = current_state['position']
        target_pos = np.array(target_state['position'], dtype=np.float32)
        position_error = target_pos - current_pos
        
        if not hasattr(self, 'integral_error'):
            self.integral_error = np.zeros(3, dtype=np.float32)
            self.last_error = np.zeros(3, dtype=np.float32)
        self.integral_error += position_error * self.dt
        derivative_error = (position_error - self.last_error) / self.dt
        self.last_error = position_error.copy()
        control = (kp * position_error + 
                  ki * self.integral_error + 
                  kd * derivative_error)
        
        joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        scale = 0.5
        joint_velocities[:3] = control * scale
        
        return joint_velocities
    
    def run_simulation(self, demo_type: str, controller_params: Dict[str, float], 
                      duration: float = 10.0) -> Dict[str, List[float]]:
        demo_config = self.setup_control_system_demo(demo_type)
        
        time_points = []
        positions = []
        errors = []
        control_signals = []
        
        start_time = time.time()
        current_time = 0
        
        while current_time < duration:
            current_state = {
                'position': self.get_end_effector_position(),
                'velocity': self.get_joint_velocities()
            }
            if demo_type == "trajectory_tracking":
                target_position = demo_config['trajectory'](current_time)
            else:
                target_position = demo_config['target']
                
            target_state = {
                'position': target_position,
                'velocity': np.zeros(self.num_joints)
            }
            control_signal = self.apply_control(current_state, target_state, controller_params)
            if demo_type == "disturbance_rejection":
                disturbance = demo_config['disturbance'](current_time)
                control_signal[:len(disturbance)] += disturbance
            
            self.set_joint_velocities(control_signal)
            self.scene.step()
            #store data
            time_points.append(current_time)
            positions.append(current_state['position'])
            errors.append(np.linalg.norm(np.array(target_position) - np.array(current_state['position'])))
            control_signals.append(control_signal)
            current_time = time.time() - start_time
        
        return {
            'time': time_points,
            'position': positions,
            'error': errors,
            'control': control_signals,
            'description': demo_config['description']
        }

    def open_viewer(self):
        if self.scene:
            try:
                viewer_scene = gs.Scene(
                    viewer_options = gs.options.ViewerOptions(
                        camera_pos    = (0, -3.5, 2.5),
                        camera_lookat = (0.0, 0.0, 0.5),
                        camera_fov    = 30,
                        max_FPS       = 60,
                    ),
                    sim_options = gs.options.SimOptions(
                        dt = self.dt,
                    ),
                    show_viewer = True,
                )
                
                # add ground plane
                viewer_scene.add_entity(
                    gs.morphs.Plane(),
                )
                
                # add Franka robot with same configuration
                viewer_scene.add_entity(
                    gs.morphs.MJCF(
                        file  = 'xml/franka_emika_panda/panda_simple.xml',
                        pos   = (0.0, 0.0, 0.0),
                        euler = (0, 0, 0),
                    ),
                )
                viewer_scene.build()                
                for _ in range(10):
                    viewer_scene.step()
                
                return viewer_scene
            except Exception as e:
                print(f"Error opening viewer: {e}")
                return None
        return None

def main():
    simulator = GenesisSimulator()
    controller_params = {'kp': 1.0, 'ki': 0.1, 'kd': 0.05}
    results = simulator.run_simulation("position_control", controller_params)
    
    print("Simulation complete!")
    print(f"Final position error: {results['error'][-1]:.3f}")

if __name__ == "__main__":
    main()
