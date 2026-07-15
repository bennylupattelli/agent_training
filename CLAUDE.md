# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General Points
- You are allowed to say 'I don't know'
- Use direct quotes for factual grounding
- Verify claims with citations

## Project Overview
This repo is concerned with batch training and inference of Unity MLAgents models in the 'Octagon' game.
Two other local git repos are important for understanding this project. Octagon is the repo for the Unity implementation
of the Octagon game, both for MLAgents agents and for human players. Octagon_analysis is the repo for analysis, plotting,
and visualisation of the data logged in the Octagon game by both human players and MLAgents RL agents. These repos can be found
in the 'Related Repos' section, and should always be accessible and kept in context.

## Related Repos
- '/home/tom/Unity/Octagon' - Unity project for the Octagon game. Contains the game logic, as well as implementing
netcode for human players (to allow two players to interact across machines), and the code for agent training and inference, which
does not use any netcode and is always local.
- '/home/tom/repos/octagon_analysis' - Python analysis repository for all behavioural and statistical analysis for Unity Octagon,
as well as visualisation and plotting.