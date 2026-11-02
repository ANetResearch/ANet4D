"""Ambient population backend (route A): open-source solvers for the motion of the scene's targets.

Vehicles are simulated by Eclipse SUMO on a network compiled from the measured vehicle support (awr.life.ambient.roads);
pedestrians by JuPedSim on the measured walkable ground outside the carriageway (awr.life.ambient.walkway). This package
only composes the stage, the activities of the targets, the coupling of the two solvers at crossings and the
conversion to the D3 tape; it contains no motion model of its own.
"""
