# Lego Sorter Project

A software control system for an automated LEGO sorting machine inspired by Amazon's robotic warehouses.

## Core Concept: Chaotic Storage
Unlike traditional sorters that separate bricks by type into specific bins, this system uses "Chaotic Storage".
- **No Fixed Bins**: Any box can hold any mix of bricks.
- **High Density**: Each box holds exactly 30 bricks.
- **Database Tracking**: The system knows exactly which box contains which specific brick.

## System Overview
The machine operates in two main modes:

1.  **Fill Cycle**:
    - Bricks are fed via conveyor.
    - Webcam identifies the brick.
    - Machine dispenses the brick into the current active "Fill Box".
    - Database records `{BrickID, BoxID}`.

2.  **Retrieval Stage**:
    - User requests a Part (e.g., "2x4 Red Brick").
    - System queries DB for the nearest box containing that part.
    - System retrieves the box (or directs user to it).
    - Database entry is removed/decremented upon retrieval.

## Getting Started
See [BACKGROUND.md](./BACKGROUND.md) for detailed operational logic.
See [ARCHITECTURE.md](./ARCHITECTURE.md) for system design.
