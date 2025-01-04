import openai
import numpy as np
import plotly.graph_objects as go
import io
import base64

class SimulationInterpreter:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            openai.api_key = api_key
            
    def interpret_simulation(self, simulation_data, user_question):
        if not self.api_key:
            return "Please set up your OpenAI API key first."
        
        context = self._prepare_simulation_context(simulation_data)
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": """You are an expert control systems engineering tutor. 
                    Analyze the simulation data and provide clear, technical explanations about the system's behavior,
                    performance metrics, and control theory concepts."""},
                    {"role": "user", "content": f"Given this simulation data: {context}\n\nUser question: {user_question}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error interpreting simulation: {str(e)}"

    def interpret_root_locus(self, root_locus_data, transfer_function, user_question):
        if not self.api_key:
            return "Please set up your OpenAI API key first."
            
        context = self._prepare_root_locus_context(root_locus_data, transfer_function)
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": """You are an expert control systems engineering tutor.
                    Analyze the root locus plot and transfer function to explain system stability, 
                    performance characteristics, and control theory concepts."""},
                    {"role": "user", "content": f"Given this root locus data: {context}\n\nUser question: {user_question}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error interpreting root locus: {str(e)}"

    def _prepare_simulation_context(self, simulation_data):
        if 'message' in simulation_data:
            return simulation_data['message']
            
        context = {
            'Initial State': {
                'Position': simulation_data['initial_state']['position'],
                'Joint Positions': simulation_data['initial_state']['joint_positions']
            },
            'Final State': {
                'Position': simulation_data['final_state']['position'],
                'Joint Positions': simulation_data['final_state']['joint_positions']
            },
            'Trajectory Summary': {
                'Number of Points': len(simulation_data['trajectory']),
                'Start Position': simulation_data['trajectory'][0],
                'End Position': simulation_data['trajectory'][-1],
                'Total Time': simulation_data['time'][-1]
            },
            'Performance': {
                'Final Error Values': simulation_data['error'],
                'Controller': simulation_data['controller_params'],
                'Demo Type': simulation_data['demo_type']
            },
            'Description': simulation_data['description']
        }
        return str(context)

    def _prepare_root_locus_context(self, root_locus_data, transfer_function):
        context = {
            "transfer_function": str(transfer_function),
            "poles": root_locus_data.get("poles", []),
            "zeros": root_locus_data.get("zeros", []),
            "gain": root_locus_data.get("gain", 0),
            "stability_margin": root_locus_data.get("stability_margin", 0)
        }
        return str(context)
