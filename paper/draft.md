Title

Learning-Based Autonomous Navigation in Dynamic Environments: A Comparative Study

Abstract

Autonomous navigation in dynamic environments remains a core challenge in robotics.
This work presents a comparative study between classical path planning (A*) and reinforcement learning–based navigation using a custom-built simulation environment with dynamic obstacles. A Q-learning agent is evaluated under partial observability and compared against an optimal planner under varying environmental conditions. Results highlight the strengths and limitations of learning-based approaches in non-static settings.

1. Introduction

Autonomous navigation is fundamental in robotics

Classical planners assume static worlds

Learning-based methods adapt to dynamics

Simulation is the standard first validation step

2. Related Work

Classical path planning (A*)

Reinforcement learning in robotics

Navigation under dynamic obstacles

Simulation-to-real workflows

(You can cite textbooks / surveys later)

3. Environment Design

2D grid-world simulator

Static + dynamic obstacles

Partial observability via local sensor window

Episode-based formulation

4. Methods
4.1 A* Planner

Optimal shortest-path algorithm

Operates on static map

No adaptation to dynamic obstacles

4.2 Q-Learning Agent

Tabular reinforcement learning

ε-greedy exploration

Reward shaping

Learns policy through interaction

5. Experimental Setup

Grid size: 10×10

Obstacle density: variable

Metrics:

Success rate

Collision rate

Episode length

6. Results

A* performs optimally in static environments

Performance degrades with dynamic obstacles

Q-learning adapts better to dynamics

Trade-off between optimality and adaptability

7. Discussion

Learning-based methods handle uncertainty better

Classical planners remain strong baselines

Partial observability increases realism

Simulation validates feasibility before real robots

8. Conclusion

This study demonstrates that reinforcement learning provides a viable alternative to classical planning in dynamic environments, particularly when full observability cannot be assumed. Future work includes deep reinforcement learning and real-world robotic deployment.

9. Future Work

Deep Q-Networks

Continuous action spaces

ROS 2 integration

Sim-to-real transfer