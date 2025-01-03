import streamlit as st
import numpy as np
import control
import plotly.graph_objects as go
from tutor_agent import ControlSystemsTutor, RootLocusSimulator
from genesis_simulator import GenesisSimulator
import io
import platform

def create_3d_visualization(positions):
    positions = np.array(positions)
    fig = go.Figure()
    frames = []
    for i in range(1, len(positions) + 1):
        frame_data = [
            go.Scatter3d(
                x=[positions[0, 0]],
                y=[positions[0, 1]],
                z=[positions[0, 2]],
                mode='markers',
                name='Start Position',
                marker=dict(color='red', size=10, symbol='circle'),
                showlegend=True if i == len(positions) else False
            ),
            go.Scatter3d(
                x=positions[:i, 0],
                y=positions[:i, 1],
                z=positions[:i, 2],
                mode='lines',
                name='Trajectory',
                line=dict(color='blue', width=2),
                showlegend=True if i == len(positions) else False
            ),
            go.Scatter3d(
                x=[positions[i-1, 0]],
                y=[positions[i-1, 1]],
                z=[positions[i-1, 2]],
                mode='markers',
                name='Current Position',
                marker=dict(color='yellow', size=8, symbol='circle'),
                showlegend=True if i == len(positions) else False
            ),
            go.Scatter3d(
                x=[positions[-1, 0]],
                y=[positions[-1, 1]],
                z=[positions[-1, 2]],
                mode='markers',
                name='Target Position',
                marker=dict(color='green', size=10, symbol='circle'),
                showlegend=True if i == len(positions) else False
            )
        ]
        frames.append(go.Frame(data=frame_data, name=f'frame{i}'))
    
    # initial state
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]],
        y=[positions[0, 1]],
        z=[positions[0, 2]],
        mode='markers',
        name='Start Position',
        marker=dict(color='red', size=10, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[positions[-1, 0]],
        y=[positions[-1, 1]],
        z=[positions[-1, 2]],
        mode='markers',
        name='Target Position',
        marker=dict(color='green', size=10, symbol='circle')
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position',
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=600,
        height=600,
        title='Robot End-Effector Trajectory',
        showlegend=True,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '},
            'steps': [
                {
                    'label': f'{i+1}',
                    'method': 'animate',
                    'args': [[f'frame{i+1}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                for i in range(len(frames))
            ]
        }]
    )
    fig.frames = frames
    return fig

def create_interactive_sandbox():
    st.title("Control Systems Engineering Sandbox")
    tutor = ControlSystemsTutor()
    root_locus_sim = RootLocusSimulator()
    genesis_sim = GenesisSimulator()
    tutor.current_topic = 'root_locus'
    st.sidebar.header("Learning Mode")
    mode = st.sidebar.radio(
        "Select Mode",
        ["Theory Practice", "Physical Simulation", "Free Exploration"]
    )
    
    if mode == "Theory Practice":
        difficulty = st.sidebar.selectbox(
            "Select Difficulty",
            ["Basic", "Intermediate", "Advanced"]
        )
        
        if st.sidebar.button("Generate New Problem"):
            problem = tutor.generate_practice_problem(difficulty.lower())
            
            if 'error' not in problem:
                st.session_state['current_problem'] = problem
                st.session_state['show_solution'] = False
                st.session_state['show_hints'] = False
            else:
                st.error(problem['error'])
        
        if 'current_problem' in st.session_state:
            problem = st.session_state['current_problem']
            st.subheader("Problem")
            st.write(problem['question'])
            fig = root_locus_sim.plot_root_locus(problem['num'], problem['den'])
            st.plotly_chart(fig)
            if st.button("Show Hints"):
                st.session_state['show_hints'] = not st.session_state.get('show_hints', False)
            
            if st.button("Show Solution"):
                st.session_state['show_solution'] = not st.session_state.get('show_solution', False)
            
            if st.session_state.get('show_hints', False):
                st.subheader("Hints")
                for hint in problem['hints']:
                    st.write(f"• {hint}")
            
            if st.session_state.get('show_solution', False):
                st.subheader("Solution")
                for key, value in problem['solution'].items():
                    st.write(f"**{key.title()}:** {value}")
    
    elif mode == "Physical Simulation":
        st.subheader("Physical Control System Simulation")
        st.write("""
        Explore different control scenarios by adjusting PID gains and observing the robot's behavior.
        explore the sections below to learn more about each control scenario.
        """)
        
        with st.expander("🎯 Position Control Info"):
            st.write("""
            ### Position Control
            
            In position control, the goal is to move the robot to a specific target position and maintain it there.
            
            **Key Aspects:**
            - **Steady-State Error**: The difference between target and final position
            - **Settling Time**: Time taken to reach and stay within ±2% of target
            - **Overshoot**: Maximum deviation beyond the target
            
            **Tuning Tips:**
            1. Start with P control only (Ki=Kd=0)
            2. Increase P until you get reasonable response speed
            3. Add D to reduce overshoot
            4. Add small I to eliminate steady-state error
            
            **Common Issues:**
            - High P: Oscillations and overshoot
            - Low P: Slow response and steady-state error
            - High D: Noisy response
            - High I: Integral windup and oscillations
            """)
            
        with st.expander("📈 Trajectory Tracking Info"):
            st.write("""
            ### Trajectory Tracking
            
            Trajectory tracking involves following a time-varying reference path, common in robotic applications.
            
            **Key Aspects:**
            - **Tracking Error**: Distance between robot and desired path
            - **Phase Lag**: Delay between command and response
            - **Corner Smoothing**: Behavior at sharp trajectory changes
            
            **Tuning Tips:**
            1. Higher gains needed compared to position control
            2. D term crucial for anticipating trajectory changes
            3. Balance between accuracy and smooth motion
            
            **Advanced Techniques:**
            - Feedforward control for known trajectories
            - Path prediction and lookahead
            - Velocity profiling for smooth acceleration
            """)
            
        with st.expander("🌪️ Disturbance Rejection Info"):
            st.write("""
            ### Disturbance Rejection
            
            Testing the controller's ability to maintain position or trajectory despite external forces.
            
            **Types of Disturbances:**
            - **Impulse**: Sudden, short-duration forces
            - **Step**: Constant forces (like payload changes)
            - **Periodic**: Oscillating disturbances
            
            **Performance Metrics:**
            - **Maximum Deviation**: Peak error during disturbance
            - **Recovery Time**: Time to return to target
            - **Steady-State Recovery**: Final error after disturbance
            
            **Tuning Strategy:**
            1. Integral term crucial for constant disturbances
            2. Derivative term helps quick recovery
            3. Consider disturbance observers for better rejection
            """)
        
        st.write("### Simulation Parameters")
        demo_type = st.selectbox(
            "Select Demo",
            ["position_control", "trajectory_tracking", "disturbance_rejection"]
        )
        st.sidebar.subheader("Controller Parameters")
        kp = st.sidebar.slider("Proportional Gain (Kp)", 0.0, 5.0, 1.0)
        ki = st.sidebar.slider("Integral Gain (Ki)", 0.0, 2.0, 0.1)
        kd = st.sidebar.slider("Derivative Gain (Kd)", 0.0, 2.0, 0.05)
        
        controller_params = {'kp': kp, 'ki': ki, 'kd': kd}
        st.info("""
        💡 The 3D visualization shows the robot's end-effector trajectory:
        
        - Blue line: Complete movement path
        - Red dot: Final position
        
        Interactive controls:
        - Rotate: Click and drag
        - Zoom: Scroll or pinch
        - Pan: Right-click and drag
        - Reset view: Double click
        
        You can also:
        - Hover over points for coordinates
        - Click legend items to show/hide
        """)
        
        if st.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                results = genesis_sim.run_simulation(demo_type, controller_params)
                st.subheader("3D Robot Trajectory")
                positions = np.array(results['position'])
                fig_3d = create_3d_visualization(positions)
                st.plotly_chart(fig_3d)
                st.subheader("Position vs Time")
                fig_pos = go.Figure()
                fig_pos.add_trace(go.Scatter(
                    x=results['time'],
                    y=positions[:, 0],
                    name='X Position'
                ))
                fig_pos.add_trace(go.Scatter(
                    x=results['time'],
                    y=positions[:, 1],
                    name='Y Position'
                ))
                fig_pos.update_layout(
                    xaxis_title='Time (s)',
                    yaxis_title='Position'
                )
                st.plotly_chart(fig_pos)
                st.subheader("Control Performance")
                fig_error = go.Figure()
                fig_error.add_trace(go.Scatter(
                    x=results['time'],
                    y=results['error'],
                    name='Position Error'
                ))
                fig_error.update_layout(
                    xaxis_title='Time (s)',
                    yaxis_title='Error'
                )
                st.plotly_chart(fig_error)
                st.subheader("Demo Description")
                st.write(results['description'])
    
    else:
        st.subheader("Transfer Function Builder")
        st.write("""
        Enter coefficients for your transfer function G(s) = Num(s)/Den(s).
        The coefficients should be in descending order of powers of s.
        """)
        
        st.write("Numerator coefficients (highest to lowest power)")
        num = st.text_input("Enter coefficients separated by comma", "1")
        num = [float(x.strip()) for x in num.split(",")]
        
        st.write("Denominator coefficients (highest to lowest power)")
        den = st.text_input("Enter coefficients separated by comma", "1,2,1")
        den = [float(x.strip()) for x in den.split(",")]
        
        # Add example systems with explanations
        with st.expander("Click to see example transfer functions"):
            st.info("""
            Try these interesting transfer functions:
            
            1. Current system (double pole): 
               - Num = [1], Den = [1,2,1]  (1/(s² + 2s + 1))
               - Shows root locus for a system with repeated real poles at s = -1
            
            2. Complex poles: 
               - Num = [1], Den = [1,2,5]  (1/(s² + 2s + 5))
               - Shows how poles move from complex locations (-1 ± 2j)
            
            3. System with zero: 
               - Num = [1,2], Den = [1,3,2]  ((s + 2)/(s² + 3s + 2))
               - Demonstrates zero's effect on root locus path
            
            4. Third-order system: 
               - Num = [1], Den = [1,6,11,6]  (1/(s + 1)(s + 2)(s + 3))
               - Shows breakaway points and asymptotes
            
            5. Marginally stable: 
               - Num = [1], Den = [1,0,1]  (1/(s² + 1))
               - Shows poles on imaginary axis (±j)
            """)
        if 'root_locus_generated' not in st.session_state:
            st.session_state.root_locus_generated = False
            
        if st.button("Generate Root Locus") or st.session_state.root_locus_generated:
            st.session_state.root_locus_generated = True
            try:
                plot_container = st.container()
                st.subheader("System Analysis")
                st.write("""
                Adjust the gain K to analyze the closed-loop system stability and performance.
                The purple stars show the current closed-loop pole locations.
                
                The analysis shows:
                - Stability status
                - Pole locations
                - Time-domain characteristics (settling time, overshoot, etc.)
                """)
                
                K = st.slider("System Gain (K)", 0.0, 10.0, 1.0, 0.1, key="gain_slider")
                with plot_container:
                    fig = root_locus_sim.plot_root_locus(num, den, K)
                    st.plotly_chart(fig, key="root_locus_plot", use_container_width=True)
                
                # stability analysis
                stability = root_locus_sim.analyze_stability(num, den, K)
                st.code(stability, language="text")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

def main():
    create_interactive_sandbox()

if __name__ == "__main__":
    main()
