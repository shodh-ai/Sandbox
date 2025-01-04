from flask import Flask, jsonify, request
from genesis_simulator import GenesisSimulator
import numpy as np

app = Flask(__name__)

simulator = GenesisSimulator()

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        # get simulation parameters
        data = request.get_json()
        demo_type = data.get('demo_type', 'position_control')
        target_position = data.get('target_position', [0.5, 0.5, 0.5])
        initial_position = data.get('initial_position', [0, 0, 0])
        controller_params = data.get('controller_params', {
            'kp': 1.0,
            'ki': 0.1,
            'kd': 0.2
        })
        
        # run simulation
        results = simulator.run_simulation(
            demo_type=demo_type,
            target_position=target_position,
            initial_position=initial_position,
            controller_params=controller_params
        )
        
        # convert for JSON serialization
        response_data = {
            'initial_state': {
                'position': initial_position,
                'joint_positions': results['initial_state']['joint_positions'].tolist() if isinstance(results['initial_state']['joint_positions'], np.ndarray) else results['initial_state']['joint_positions']
            },
            'final_state': {
                'position': results['final_state']['position'].tolist() if isinstance(results['final_state']['position'], np.ndarray) else results['final_state']['position'],
                'joint_positions': results['final_state']['joint_positions'].tolist() if isinstance(results['final_state']['joint_positions'], np.ndarray) else results['final_state']['joint_positions']
            },
            'trajectory': results['trajectory'].tolist() if isinstance(results['trajectory'], np.ndarray) else results['trajectory'],
            'time': results['time'].tolist() if isinstance(results['time'], np.ndarray) else results['time'],
            'error': results['error'].tolist() if isinstance(results['error'], np.ndarray) else results['error'],
            'controller_params': results['controller_params'],
            'demo_type': results['demo_type'],
            'description': results['description']
        }
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/simulation_config', methods=['GET'])
def get_simulation_config():
    """Get available simulation configurations and parameters"""
    return jsonify({
        'demo_types': ['position_control', 'trajectory_tracking'],
        'parameter_ranges': {
            'kp': {'min': 0.1, 'max': 5.0, 'default': 1.0},
            'ki': {'min': 0.0, 'max': 2.0, 'default': 0.1},
            'kd': {'min': 0.0, 'max': 2.0, 'default': 0.2}
        },
        'position_limits': {
            'x': {'min': -1.0, 'max': 1.0},
            'y': {'min': -1.0, 'max': 1.0},
            'z': {'min': -1.0, 'max': 1.0}
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
