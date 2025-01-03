import sympy as sp
from typing import Dict, List, Any
import numpy as np
import control
import plotly.graph_objects as go

class ControlSystemsTutor:
    def __init__(self):
        self.student_progress = {
            'understanding': 0,
            'memory': 0,
            'critical_thinking': 0,
            'applied_knowledge': 0
        }
        self.current_topic = None
        
    def introduce_topic(self, topic: str) -> str:
        self.current_topic = topic
        topics = {
            'root_locus': """
            Let's explore Root Locus! This is a powerful graphical method that shows how the poles 
            of a closed-loop system change as we vary a parameter (usually the gain K).
            
            Key concepts we'll cover:
            1. Poles and zeros
            2. System stability
            3. Drawing rules
            4. Effect on system response
            
            Would you like to start with theory or jump into a practical example?
            """
        }
        return topics.get(topic, "Topic not found in curriculum")
    
    def evaluate_understanding(self, student_response: Dict[str, Any]) -> Dict[str, float]:
        # Implementation for evaluating student responses
        pass
    
    def generate_practice_problem(self, difficulty: str) -> Dict[str, Any]:
        if self.current_topic == 'root_locus':
            problems = {
                'basic': {
                    'num': [1],
                    'den': [1, 2, 1],  # (s + 1)(s + 1)
                    'question': """Analyze this second-order system with transfer function G(s) = 1/(s² + 2s + 1):
                    1. Find the poles of the system
                    2. What is the breakaway point (if any)?
                    3. Is the system stable for K > 0?""",
                    'hints': [
                        "Start by factoring the denominator",
                        "Remember that poles are roots of the denominator",
                        "For stability, check if poles are in left half-plane"
                    ],
                    'solution': {
                        'poles': [-1, -1],
                        'breakaway': -1,
                        'stability': "Stable for all K > 0"
                    }
                },
                'intermediate': {
                    'num': [1],
                    'den': [1, 0, 1],  # s² + 1
                    'question': """Analyze this system with transfer function G(s) = 1/(s² + 1):
                    1. Find the poles of the system
                    2. Sketch the root locus
                    3. For what values of K is the system stable?""",
                    'hints': [
                        "The poles are imaginary",
                        "Root locus branches start at poles",
                        "Consider where branches cross imaginary axis"
                    ],
                    'solution': {
                        'poles': [1j, -1j],
                        'stability': "System is marginally stable at K=0, unstable for K>0"
                    }
                },
                'advanced': {
                    'num': [1, 0],  # s
                    'den': [1, 2, 2, 0],  # s(s² + 2s + 2)
                    'question': """Analyze this system with transfer function G(s) = s/(s³ + 2s² + 2s):
                    1. Find all poles and zeros
                    2. Determine the asymptotes
                    3. Find the breakaway points
                    4. What is the critical gain Kcr?""",
                    'hints': [
                        "Factor out s from denominator",
                        "Use root locus rules for asymptotes",
                        "Use derivative method for breakaway points"
                    ],
                    'solution': {
                        'poles': [0, -1 + 1j, -1 - 1j],
                        'zeros': [0],
                        'asymptotes': "Centered at σ = -1, angles: ±90°"
                    }
                }
            }
            
            if difficulty.lower() in problems:
                problem = problems[difficulty.lower()]
                # Create transfer function for visualization
                sys = control.TransferFunction(problem['num'], problem['den'])
                
                return {
                    'transfer_function': sys,
                    'question': problem['question'],
                    'hints': problem['hints'],
                    'solution': problem['solution'],
                    'num': problem['num'],
                    'den': problem['den'],
                    'difficulty': difficulty
                }
            
        return {
            'error': 'Invalid difficulty level or topic. Choose from: basic, intermediate, advanced'
        }
    
class RootLocusSimulator:
    def __init__(self):
        self.s = sp.Symbol('s')
        
    def plot_root_locus(self, num: List[float], den: List[float], K: float = None):
        sys = control.TransferFunction(num, den)
        # root locus points
        rlist, klist = control.rlocus(sys, plot=False)
        
        fig = go.Figure()
        
        for i in range(rlist.shape[1]):
            fig.add_trace(go.Scatter(
                x=np.real(rlist[:, i]),
                y=np.imag(rlist[:, i]),
                mode='lines',
                name='Root Locus',
                line=dict(color='rgba(0,0,255,0.6)', width=2),
                showlegend=(i == 0)
            ))
        
        poles = control.poles(sys)
        zeros = control.zeros(sys)
        points = []
        if len(poles) > 0:
            points.extend(poles)
            fig.add_trace(go.Scatter(
                x=np.real(poles),
                y=np.imag(poles),
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='red',
                    line=dict(width=2)
                ),
                name='Open-Loop Poles'
            ))
            
        if len(zeros) > 0:
            points.extend(zeros)
            fig.add_trace(go.Scatter(
                x=np.real(zeros),
                y=np.imag(zeros),
                mode='markers',
                marker=dict(
                    symbol='circle-open',
                    size=12,
                    color='green',
                    line=dict(width=2)
                ),
                name='Zeros'
            ))
            
        # add closed-loop poles if K is provided
        if K is not None:
            closed_sys = control.feedback(sys * K, 1)
            cl_poles = control.poles(closed_sys)
            points.extend(cl_poles)
            
            # split real and complex poles
            real_poles = [p for p in cl_poles if np.imag(p) == 0]
            complex_poles = [p for p in cl_poles if np.imag(p) != 0]
            
            if real_poles:
                fig.add_trace(go.Scatter(
                    x=np.real(real_poles),
                    y=np.imag(real_poles),
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='purple',
                        line=dict(width=2)
                    ),
                    name=f'Closed-Loop Poles (K={K:.2f})'
                ))
            
            if complex_poles:
                fig.add_trace(go.Scatter(
                    x=np.real(complex_poles),
                    y=np.imag(complex_poles),
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='purple',
                        line=dict(width=2)
                    ),
                    name=f'Closed-Loop Poles (K={K:.2f})',
                    showlegend=len(real_poles) == 0  # Only show legend if no real poles
                ))
        
        # imaginary axis
        fig.add_shape(
            type="line",
            x0=0, y0=-10, x1=0, y1=10,
            line=dict(color="gray", dash="dash", width=1)
        )
        
        if points:
            max_real = max(abs(np.real(points))) if points else 5
            max_imag = max(abs(np.imag(points))) if points else 5
            axis_limit = max(max_real, max_imag) * 1.5
        else:
            axis_limit = 5
            
        fig.update_layout(
            title='Root Locus Plot',
            xaxis_title='Real Axis',
            yaxis_title='Imaginary Axis',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            xaxis=dict(
                range=[-axis_limit, axis_limit],
                zeroline=True,
                gridcolor='lightgray',
                zerolinecolor='black',
                zerolinewidth=2,
                dtick=1
            ),
            yaxis=dict(
                range=[-axis_limit, axis_limit],
                zeroline=True,
                gridcolor='lightgray',
                zerolinecolor='black',
                zerolinewidth=2,
                dtick=1,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='white',
            width=700,
            height=700
        )
        
        return fig
    
    def analyze_stability(self, num: List[float], den: List[float], K: float) -> str:
        # transfer functions
        sys = control.TransferFunction(num, den)
        closed_sys = control.feedback(sys * K, 1)
        poles = control.poles(closed_sys)
        
        analysis = []
        # stability check
        if all(np.real(poles) < 0):
            analysis.append(" System is stable")
            
            # damping ratios and natural frequencies
            for pole in poles:
                if np.imag(pole) != 0:
                    zeta = -np.real(pole) / np.sqrt(np.real(pole)**2 + np.imag(pole)**2)
                    wn = np.sqrt(np.real(pole)**2 + np.imag(pole)**2)
                    
                    Ts = 4.0 / (-np.real(pole))
                    Tp = np.pi / np.abs(np.imag(pole))
                    PO = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
                    
                    analysis.append(f"Complex poles at {np.real(pole):.2f} ± {np.imag(pole):.2f}j")
                    analysis.append(f"• Damping ratio (ζ) = {zeta:.3f}")
                    analysis.append(f"• Natural frequency (ωn) = {wn:.2f} rad/s")
                    analysis.append(f"• Settling time (Ts) ≈ {Ts:.2f} sec")
                    analysis.append(f"• Peak time (Tp) ≈ {Tp:.2f} sec")
                    analysis.append(f"• Percent overshoot ≈ {PO:.1f}%")
                else:
                    Ts = 4.0 / (-np.real(pole))  # Settling time
                    analysis.append(f"Real pole at {np.real(pole):.2f}")
                    analysis.append(f"• Time constant (τ) = {-1/np.real(pole):.2f} sec")
                    analysis.append(f"• Settling time (Ts) ≈ {Ts:.2f} sec")
        else:
            analysis.append(" System is unstable")
            unstable_poles = [p for p in poles if np.real(p) >= 0]
            analysis.append(f"Unstable poles found at:")
            for pole in unstable_poles:
                if np.imag(pole) != 0:
                    analysis.append(f"• {np.real(pole):.2f} ± {np.imag(pole):.2f}j")
                else:
                    analysis.append(f"• {np.real(pole):.2f}")
        
        return "\n".join(analysis)

def main():
    tutor = ControlSystemsTutor()
    simulator = RootLocusSimulator()
    print(tutor.introduce_topic('root_locus'))
    
    # transfer function: G(s) = 1/(s^2 + 2s + 1)
    num = [1]
    den = [1, 2, 1]
    plot = simulator.plot_root_locus(num, den, K=1.0)
    # plot.show()
    stability = simulator.analyze_stability(num, den, K=1.0)
    print(f"\nStability Analysis: {stability}")

if __name__ == "__main__":
    main()
