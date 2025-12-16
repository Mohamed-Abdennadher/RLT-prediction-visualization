# Architecture & Data Flow

This document explains how the different components work together.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                        │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PYTHON BACKEND                             │
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │  Data Input  │ ───▶ │  RLT Builder │ ───▶ │   Exporter   │  │
│  │   (X, y)     │      │  (Algorithm) │      │  (to JSON)   │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │         │
│         │                      │                      ▼         │
│         ▼                      ▼              tree_data.json    │
│   make_regression()     fit_recursive()      training_history  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   JSON DATA TRANSFER                            │
│                                                                 │
│   Move files to: my-react-app/src/                             │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     REACT FRONTEND                              │
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   App.jsx    │ ───▶ │  RLTTree OR  │ ───▶ │  RLTNode.jsx │  │
│  │  (Router)    │      │TrainingPlayer│      │ (Renderer)   │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │         │
│         ▼                      ▼                      ▼         │
│   Load JSON          React Flow Library       Recharts         │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BROWSER DISPLAY                            │
│                  Interactive Tree Visualization                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Python Component Flow

### Building a Single Tree

```
START
  │
  ├─▶ Create RLTBuilder instance
  │   └─ Configure: reinforcement, muting_rate, nmin, etc.
  │
  ├─▶ Call fit(X, y, sample_indices)
  │   └─ Start recursive building
  │
  ├─▶ _fit_recursive() [ROOT NODE]
  │   │
  │   ├─ STEP 1: Initialize Node
  │   │   └─ Assign unique ID, log NODE_INIT event
  │   │
  │   ├─ STEP 2: Check Stopping Criteria
  │   │   ├─ n_samples < nmin? ───▶ Make Leaf
  │   │   ├─ Pure node? ──────────▶ Make Leaf
  │   │   └─ depth > max? ────────▶ Make Leaf
  │   │
  │   ├─ STEP 3: Reinforcement (if enabled)
  │   │   ├─ Run pilot Random Forest
  │   │   │   └─ Log PILOT_RUN event
  │   │   ├─ Calculate feature importances
  │   │   ├─ Sort features by importance
  │   │   ├─ Mute bottom X% of features
  │   │   │   └─ Log MUTE_VARS event
  │   │   └─ Update candidate pool
  │   │
  │   ├─ STEP 4: Find Best Split
  │   │   ├─ Try random features from candidate pool
  │   │   ├─ Try random thresholds
  │   │   ├─ Calculate variance reduction
  │   │   ├─ Select best (feature, threshold) pair
  │   │   └─ Log SPLIT_DECISION event
  │   │
  │   ├─ STEP 5: Recurse to Children
  │   │   ├─ Split samples: left (≤ threshold), right (> threshold)
  │   │   ├─ Call _fit_recursive() for LEFT child
  │   │   │   └─ Pass updated muted_set
  │   │   ├─ Call _fit_recursive() for RIGHT child
  │   │   │   └─ Pass updated muted_set
  │   │   └─ Attach children to current node
  │   │
  │   └─▶ Return RLTNode
  │
  └─▶ Return (root_node, history_log)
  │
END
```

### Make Leaf Node

```
_make_leaf()
  │
  ├─▶ Calculate prediction
  │   ├─ Regression: mean(y_curr)
  │   └─ Classification: mode(y_curr)
  │
  ├─▶ Log MAKE_LEAF event
  │
  └─▶ Return RLTNode(is_leaf=True, prediction=value)
```

---

## React Component Flow

### Static Tree View (RLTTree)

```
RLTTree Component
  │
  ├─▶ Load tree_data.json
  │   └─ Contains: nodes[], edges[]
  │
  ├─▶ Call getLayoutedElements()
  │   ├─ Create Dagre graph
  │   ├─ Set graph direction (Top-to-Bottom)
  │   ├─ Add all nodes to graph
  │   ├─ Add all edges to graph
  │   ├─ Run Dagre layout algorithm
  │   └─ Return positioned nodes
  │
  ├─▶ Pass to ReactFlow
  │   ├─ nodes (with positions)
  │   ├─ edges
  │   └─ nodeTypes: { rltNode: RLTNode }
  │
  └─▶ Render
      ├─ Background grid
      ├─ Pan/zoom controls
      ├─ Minimap
      └─ Custom RLTNode for each node
```

### Animated Training Player (TrainingPlayer)

```
TrainingPlayer Component
  │
  ├─▶ Load training_history.json
  │   └─ Array of events: [{type, nodeId, data}, ...]
  │
  ├─▶ State Management
  │   ├─ currentStep: 0
  │   ├─ isPlaying: false
  │   ├─ nodes: []
  │   └─ edges: []
  │
  ├─▶ renderStep(stepIndex)
  │   │
  │   ├─ Initialize temp arrays
  │   ├─ Loop from event[0] to event[stepIndex]
  │   │   │
  │   │   ├─ If NODE_INIT:
  │   │   │   ├─ Create new node
  │   │   │   └─ Create edge to parent
  │   │   │
  │   │   ├─ If PILOT_RUN:
  │   │   │   ├─ Find node by ID
  │   │   │   ├─ Update pilotData
  │   │   │   └─ Highlight purple
  │   │   │
  │   │   ├─ If MUTE_VARS:
  │   │   │   ├─ Find node by ID
  │   │   │   └─ Update mutedVars
  │   │   │
  │   │   ├─ If SPLIT_DECISION:
  │   │   │   ├─ Find node by ID
  │   │   │   ├─ Update label/splitFeature
  │   │   │   └─ Remove highlight
  │   │   │
  │   │   └─ If MAKE_LEAF:
  │   │       ├─ Find node by ID
  │   │       ├─ Set isLeaf = true
  │   │       └─ Update prediction
  │   │
  │   ├─▶ Call getLayout() to position nodes
  │   │
  │   ├─▶ Update React state
  │   │   ├─ setNodes(layoutedNodes)
  │   │   └─ setEdges(updatedEdges)
  │   │
  │   └─▶ Smooth camera pan
  │       └─ fitView({ duration: 800ms })
  │
  ├─▶ Play/Pause Logic
  │   └─ setInterval(() => setCurrentStep(prev => prev + 1), 600ms)
  │
  └─▶ Render
      ├─ Control Panel (play, slider, info)
      ├─ ReactFlow with current nodes/edges
      └─ Event descriptions
```

---

## Data Structure Flow

### 1. Python: RLTNode (Internal)

```python
RLTNode {
  node_id: 0,
  is_leaf: False,
  feature_indices: [3],
  threshold: 0.52,
  left: RLTNode {...},
  right: RLTNode {...},
  n_samples: 300,
  muted_vars: [1, 4, 9],
  pilot_importances: {
    "3": 0.35,
    "7": 0.22,
    "12": 0.18,
    ...
  }
}
```

### 2. Python: Export to JSON (Static Tree)

```json
{
  "nodes": [
    {
      "id": "0",
      "type": "rltNode",
      "data": {
        "isLeaf": false,
        "label": "Threshold: 0.52",
        "splitFeature": "Feature_3",
        "samples": 300,
        "pilotData": [
          {"name": "Feature_3", "value": 0.35},
          {"name": "Feature_7", "value": 0.22}
        ],
        "mutedVars": ["Feature_1", "Feature_4"]
      },
      "position": {"x": 0, "y": 0}
    }
  ],
  "edges": [...]
}
```

### 3. Python: Export to JSON (Training History)

```json
[
  {
    "step": 0,
    "type": "NODE_INIT",
    "nodeId": "0",
    "parentId": null,
    "depth": 0,
    "samples": 300
  },
  {
    "step": 1,
    "type": "PILOT_RUN",
    "nodeId": "0",
    "pilotData": [
      {"name": "Var 3", "value": 0.35}
    ]
  },
  {
    "step": 2,
    "type": "MUTE_VARS",
    "nodeId": "0",
    "mutedVars": ["Var 1", "Var 4"]
  },
  ...
]
```

### 4. React: RLTNode Props

```javascript
data = {
  isLeaf: false,
  label: "Threshold: 0.52",
  splitFeature: "Feature_3",
  samples: 300,
  pilotData: [
    {name: "Feature_3", value: 0.35},
    {name: "Feature_7", value: 0.22}
  ],
  mutedVars: ["Feature_1", "Feature_4"]
}
```

---

## Algorithm Execution Timeline

### Example: Building Root + Two Children

```
Time  Event              What Happens                    Visual Update
────  ─────              ────────────                    ─────────────
t=0   NODE_INIT (0)      Create root node                Node 0 appears
      
t=1   PILOT_RUN (0)      Run RF on 300 samples          Purple border
                         Calculate importances          Bar chart fills
                         
t=2   MUTE_VARS (0)      Mute bottom 50%                Red badges
                         
t=3   SPLIT_DECISION (0) Find best: Feature_3 ≤ 0.52   Show split label
                         
t=4   NODE_INIT (1)      Create left child              Node 1 appears
                         120 samples                    Edge 0→1 created
                         
t=5   PILOT_RUN (1)      Run RF (with muted vars)       Node 1 purple
                         
t=6   MUTE_VARS (1)      Add more mutes                 More red badges
                         (cumulative)
                         
t=7   SPLIT_DECISION (1) Split on Feature_7            Show label
                         
t=8   NODE_INIT (2)      Create right child             Node 2 appears
                         180 samples                    Edge 0→2 created
                         
...   ...                Continue recursion...           Tree grows
```

---

## Key Algorithms

### 1. Pilot Random Forest (Reinforcement)

```
Input: X_curr, y_curr, allowed_features
Output: feature_importances

ALGORITHM:
1. Filter X to only allowed features
2. Train RandomForest(n_estimators=10, max_depth=3)
3. Get feature_importances_ from trained model
4. Map back to original feature indices
5. Return importance dictionary

RESULT: Knows which features are useful at this node
```

### 2. Variable Muting

```
Input: feature_importances, muting_rate
Output: muted_features

ALGORITHM:
1. Sort features by importance (descending)
2. Calculate n_mute = len(features) * muting_rate
3. Take bottom n_mute features
4. Add to cumulative muted set
5. Return updated muted set

RESULT: Child nodes won't use these features
```

### 3. Split Finding

```
Input: X, y, candidate_features
Output: (best_feature, best_threshold)

ALGORITHM:
1. Sample 5 random features from candidates
2. For each feature:
   a. Get unique values
   b. Sample 5 random thresholds
   c. For each threshold:
      - Split data: left (≤), right (>)
      - Calculate variance reduction
      - Track if best so far
3. Return best (feature, threshold) pair

RESULT: Optimal split point found
```

### 4. Tree Layout (Dagre)

```
Input: nodes[], edges[]
Output: nodes_with_positions[]

ALGORITHM:
1. Create directed graph
2. Add all nodes with dimensions
3. Add all edges
4. Run Dagre layout (rank assignment + positioning)
5. Extract (x, y) coordinates
6. Center nodes around coordinates
7. Return positioned nodes

RESULT: Non-overlapping hierarchical layout
```

---

## Performance Considerations

### Python Side:
- **Pilot RF Complexity**: O(n_samples * log(n_samples) * n_trees)
  - Runs at EACH internal node
  - Small n_trees (10) keeps it fast
  - max_depth=3 limits pilot tree growth

- **Split Finding**: O(n_features * n_thresholds * n_samples)
  - Random sampling reduces this
  - Only considers non-muted features

### React Side:
- **Layout Calculation**: O(V + E) where V=nodes, E=edges
  - Dagre is efficient for tree structures
  - Calculated once per step

- **Rendering**: O(visible_nodes)
  - React Flow uses virtualization
  - Only renders visible nodes in viewport

---

## Extension Points

Want to customize? Here are the key extension points:

### 1. Custom Split Criterion
Edit `_find_split_random()` in Python:
```python
# Current: Variance reduction
gain = current_var - (weighted_left_var + weighted_right_var)

# Alternative: Gini impurity (classification)
gain = gini(y) - (weighted_gini_left + weighted_gini_right)

# Alternative: Information gain
gain = entropy(y) - (weighted_entropy_left + weighted_entropy_right)
```

### 2. Different Pilot Models
Edit pilot model in `_fit_recursive()`:
```python
# Current: Random Forest
est = RandomForestRegressor(n_estimators=10, max_depth=3)

# Alternative: Gradient Boosting
est = GradientBoostingRegressor(n_estimators=10, max_depth=3)

# Alternative: Linear model
est = Ridge(alpha=1.0)
```

### 3. Custom Node Appearance
Edit `RLTNode.jsx`:
```jsx
// Change colors
const leafColor = isLeaf ? '#your-color' : '#another-color';

// Add new data displays
{data.customField && <div>{data.customField}</div>}

// Different chart types
<LineChart>...</LineChart>
```

### 4. Animation Events
Edit `renderStep()` in `TrainingPlayer.jsx`:
```javascript
// Add custom event type
else if (event.type === 'CUSTOM_EVENT') {
    // Your custom logic
    tempNodes[nodeIndex].data.customField = event.customData;
}
```

---

## Summary

The system follows a clean pipeline:

1. **Generate** data (or load yours)
2. **Build** RLT tree with pilot-guided feature selection
3. **Export** tree structure + history to JSON
4. **Load** in React for visualization
5. **Display** either static or animated view

Key innovation: **Pilot models at each node adaptively select features**, creating smarter, more focused trees.

For more details, see the main [README.md](README.md).
