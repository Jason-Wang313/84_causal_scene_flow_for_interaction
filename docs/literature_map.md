        # Literature Map

        Paper: 84 causal_scene_flow_for_interaction

        Field box: 3D perception for manipulation

        Thesis: Causal Scene Flow for Interaction turns the seed bet into a mechanism: Infer which scene-flow components are caused by robot action versus passive motion.

        ## Landscape Sweep Summary
        The selector ranked records from the shared 500,000-record pool. The closest visible clusters are:
        - Natural teaching for humanoid robot via human-in-the-loop scene-motion cross-modal perception (2019)
- Aerial Manipulation with Contact-Aware Onboard Perception and Hybrid Control (2026)
- Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception (2019)
- Hey Robot, Which Way Are You Going? Nonverbal Motion Legibility Cues for Human-Robot Spatial Interaction (2021)
- Flow parsing as causal source separation: A computational model for concurrent retrieval of object and self-motion information from optic flow ()
- Learning End-to-End Scene Flow by Distilling Single Tasks Knowledge (2020)
- FlowDreamer: A RGB-D World Model with Flow-based Motion Representations for Robot Manipulation (2025)
- Peripheral Visual Cues Contribute to the Perception of Object Movement During Self-Movement (2017)
- Accuracy and Tuning of Flow Parsing for Visual Perception of Object Motion During Self-Motion (2017)
- DUEL: Depth visUal Ego-motion Learning for autonomous robot obstacle avoidance (2024)
- FlowRAM: Grounding Flow Matching Policy with Region-Aware Mamba Framework for Robotic Manipulation (2025)
- FlowHOI: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation (2026)

        ## Hidden Assumptions
        - The executed trajectory is a sufficient training target.
- Unobserved physical alternatives can be averaged into uncertainty.
- Task labels capture the mechanism that caused failure.
- A planner only needs nominal feasibility.
- Embodiment-specific contact effects are nuisance variation.

        ## Boundary
        The project avoids weak moves such as bigger models, generic uncertainty, or a benchmark-only contribution. It centers a mechanism-level change: Causal scene flow for interaction keeps action-critical alternatives explicit until a physical observation collapses them.
