# Background & Operational Logic

## The Problem
Sorting LEGO by part type requires thousands of distinct bins, which takes up massive space and requires complex machinery to route parts to specific locations.

## The Solution: Chaotic Storage
Instead of organizing by **Type**, we organize by **Space**.
We treat the storage array as a set of generic containers. We don't care *what* is in a container, as long as we know *where* it is.

### 1. The Fill Cycle (Input)
The goal is to store bricks as fast as possible without sorting them mechanically.
1.  **Singulation**: A hopper/conveyor separates bricks so they pass the camera one by one.
2.  **Identification**: A webcam captures the brick. Computer Vision compares it against a known database (e.g., Rebrickable data) to find the `PartID` and `ColorID`.
3.  **Storage**:
    - The system works with a "Current Box".
    - The brick is dropped into the Current Box.
    - DB Update: `INSERT INTO Inventory (BoxID, PartID, ColorID) VALUES (CurrentBox, DiscoveredPart, DiscoveredColor)`.
    - **Capacity Check**: If the Current Box reaches 30 items, the system moves it to storage and brings in a new empty box.

### 2. The Retrieval Stage (Output)
The goal is to build a set.
1.  **Request**: User uploads a set list or requests a specific part.
2.  **Search**:
    - The system looks for `PartID` + `ColorID` in the `Inventory` table.
    - It calculates the "Cost" (distance/time) to retrieve every box containing that part.
    - It selects the optimal Box.
3.  **Retrieval**:
    - The machine interacts with the storage system to present the box to the user.
    - User picks the part.
4.  **Decrement**: The specific item is removed from the database for that `BoxID`.

## Technical Challenges
- **Computer Vision**: differentiating between very similar parts (e.g., 1x2 vs 1x3 plate) in milliseconds.
- **Database**: Efficient querying for retrieval optimization.
- **Physical constraints**: Ensuring boxes don't overflow constraints (volume/count).
