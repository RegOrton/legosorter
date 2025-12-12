# Product Requirements Document (PRD): Chaotic Storage Lego Sorter

| **Project Name** | Lego Sorter (Chaotic Storage System) |
| :--- | :--- |
| **Version** | 1.1 |
| **Status** | DRAFT |
| **Date** | 2025-12-12 |
| **Owner** | Reg |

---

## 1. Executive Summary
The **Lego Sorter** is an automated storage and retrieval system (ASRS) designed to manage large collections of LEGO bricks. Utilizing a "Chaotic Storage" methodology similar to Amazon fulfillment centers, the system optimizes for storage density and rapid ingestion rather than categorical sorting. The system leverages Computer Vision for part identification, a relational database for inventory tracking, and a 3-axis hardware gantry/conveyor system for physical manipulation.

## 2. Problem Statement
Managing a large LEGO collection by part type requires:
1.  **Excessive Space:** Thousands of bins are needed for thousands of part types, often mostly empty.
2.  **Manual Labor:** Pre-sorting parts into specific categories is time-consuming and error-prone.
3.  **Inefficient Retrieval:** Finding a specific part in a large collection "by eye" is slow.

## 3. Goals & Success Metrics
### 3.1 Primary Goals
-   **Automate Ingestion:** Eliminate the need for manual sorting.
-   **Maximize Density:** Store 30 bricks per box regardless of type to utilize 100% of box volume.
-   **Precision Retrieval:** Locate any specific brick within the system instantly.

### 3.2 Key Performance Indicators (KPIs)
-   **Ingestion Rate:** >1 brick per 2 seconds (30 bricks/minute).
-   **Identification Accuracy:** >95% correct classification (Part ID + Color).
-   **Storage Efficiency:** 90% average fill rate per box (approx 27/30 bricks).
-   **Retrieval Latency:** System identifies the optimal retrieval box in <100ms.

## 4. User Stories

| ID | Persona | Story | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **US-01** | The Sorter | As a user, I want to dump a pile of bricks into the machine so that I don't have to sort them by hand. | Machine accepts bulk input, singulates parts, and places them into storage without intervention. |
| **US-02** | The Builder | As a user, I want to request a "Red 2x4 Brick" so that I can finish my build. | System searches DB, identifies the closest box, and delivers it to the collection point. |
| **US-03** | The Builder | As a user, I want to see a list of my total inventory so I know what I can build. | Web UI displays a searchable, filterable table of all parts currently in storage. |
| **US-04** | The Operator | As a user, I want to know when the system is jammed or full so I can intervene. | System stops motors and alerts via the Dashboard if an error occurs or storage is 100% full. |

## 5. Functional Requirements

### 5.1 Ingestion & Vision System (The Eyes)
-   **FR-VIS-01 (Singulation):** The hardware MUST separate bulk bricks into a single-file stream with a minimum gap of 10mm between parts.
-   **FR-VIS-02 (Detailed Imaging):** The Vision System MUST capture high-resolution images of the top and potentially side profiles of the brick.
-   **FR-VIS-03 (Classification):** The ML Model MUST output `PartID`, `ColorID`, and `ConfidenceScore`.
    -   *Constraint:* If Confidence < 80%, the brick is rejected/re-circulated or flagged for manual review.
-   **FR-VIS-04 (Latency):** Image processing and inference MUST complete within 500ms to match the conveyor speed.

### 5.2 Core Logic & Database (The Brain)
-   **FR-LOG-01 (Chaotic Assignment):** The logic MUST assign the current identified brick to the active "Fill Box".
-   **FR-LOG-02 (Box Capacity):** The system MUST track the item count of every box.
    -   *Rule:* Max capacity = 30 items.
    -   *Action:* When count == 30, mark box as `FULL`, trigger hardware to swap for an `EMPTY` box.
-   **FR-LOG-03 (Inventory Ledger):** The Database MUST maintain an immutable record of `(BrickID, BoxID, Timestamp)` for every insertion.
-   **FR-LOG-04 (Optimization):** The Retrieval Algorithm MUST prioritize boxes that contain the *most* requested items (if retrieving a list) or the *closest* box (if retrieving a single item).

### 5.3 Hardware Control (The Body)
-   **FR-HW-01 (Platform):** Master controller MUST be a **Raspberry Pi 4/5** utilizing GPIO.
-   **FR-HW-02 (Actuation):**
    -   **Conveyor:** Continuous rotation DC/Stepper.
    -   **Diverter:** Servo-controlled gate to accept (store) or reject (re-circulate) parts.
    -   **Gantry/Carousel:** Stepper-driven system to position boxes under the diverter.
-   **FR-HW-03 (Safety):** Emergency Stop (physical button) and Software Stop (UI button) MUST immediately cut power to motors.

### 5.4 User Interface (The Face)
-   **FR-UI-01 (Technology):** **Next.js** application running on the local network (Pi's IP address).
-   **FR-UI-02 (Live Feed):** The Dashboard MUST show the live camera feed with an overlay of the "Detected Object" bounding box and label.
-   **FR-UI-03 (Inventory Explorer):** A data grid view showing `Part Image`, `Part Name`, `Color`, `Quantity`, and `Box Locations`.
-   **FR-UI-04 (Manual Control):** "Developer Mode" UI to manually jog motors and toggle actuators for maintenance.

## 6. Non-Functional Requirements
-   **NFR-01 (Reliability):** The system should be able to run for 1 hour without a software crash.
-   **NFR-02 (Maintainability):** Codebase MUST be modular (Vision, Logic, HW drivers decoupled).
-   **NFR-03 (Portability):** The Vision and Logic systems MUST run in **Docker containers** to ensure identical behavior on Dev PC and Prod Pi.
-   **NFR-04 (Data Integrity):** The SQLite database MUST be backed up periodically (or on shutdown) to prevent inventory loss.

## 7. Risks & Mitigation
| Risk | Severity | Mitigation |
| :--- | :--- | :--- |
| Lighting changes affect recognition accuracy. | High | Enclose the imaging chamber in a shroud with controlled LED lighting. |
| Similar parts (e.g., 1x2 vs 1x3 plate) are misidentified. | Medium | Use reference object for scale or measure pixel length of the bounding box. |
| Mechanical jams in the singulator. | Medium | Implement current sensing on motors to detect stalls and auto-reverse. |
| Database corruption. | Low | Use Write-Ahead Logging (WAL) mode in SQLite; frequent backups. |

## 8. Development Roadmap (Phased)
1.  **Phase 1: Computer Vision (Dockerized)** - Creating the "Eye" that can see.
2.  **Phase 2: The Brain (API & DB)** - Creating the memory and logic.
3.  **Phase 3: The Face (UI)** - Creating the interface.
4.  **Phase 4: The Body (Hardware)** - connecting the physical components.
