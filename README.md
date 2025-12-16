# RLT Prediction & Visualization

## 📖 Table of Contents
1. [Overview](#overview)
2. [What is RLT?](#what-is-rlt)
3. [Project Structure](#project-structure)
4. [Python Implementation](#python-implementation)
5. [React Visualization](#react-visualization)
6. [How It Works](#how-it-works)
7. [Setup & Usage](#setup--usage)

---

## Overview

This project implements **Reinforcement Learning Trees (RLT)** - an advanced decision tree algorithm that uses embedded pilot models to intelligently select and mute variables during tree construction. The system includes both:

1. **Python Backend**: Core RLT algorithm with training history tracking
2. **React Frontend**: Interactive visualization showing how the tree builds step-by-step

---

## What is RLT?

**Reinforcement Learning Trees (RLT)** improve traditional decision trees by:

### Key Features:
1. **Pilot Models**: At each node, runs a small Random Forest to evaluate variable importance
2. **Variable Muting**: Automatically excludes low-importance variables from child nodes
3. **Reinforcement Learning**: Uses insights from pilot models to guide splitting decisions
4. **Adaptive Feature Selection**: Focuses on most informative features at each level

### Why RLT?
- **Better Generalization**: Reduces overfitting by muting noisy variables
- **Interpretability**: Shows which features matter at each decision point
- **Flexibility**: Works for both regression and classification tasks

---

## Project Structure

```
RLT-prediction-visualization/
│
├── RLT_for_visualization.py    # Core Python implementation
│
├── RLT_notebook/                # Research & experiments
│   ├── CSV/                     # Performance metrics
│   ├── visualization/           # Results visualizations
│   └── as_fking_zhu_says.ipynb # Jupyter notebook
│
└── my-react-app/                # React visualization app
    ├── src/
    │   ├── App.jsx              # Main app entry point
    │   ├── RLTNode.jsx          # Custom node visualization
    │   ├── RLTTree.jsx          # Static tree viewer
    │   ├── TrainingPlayer.jsx   # Animated training process
    │   ├── tree_data.json       # Final tree structure
    │   └── training_history.json # Step-by-step build log
    │
    └── package.json
```

---

## Python Implementation

### File: `RLT_for_visualization.py`

This file contains the complete RLT algorithm implementation. Let's break it down:

### 1. Data Structures

```python
@dataclass
class RLTNode:
    node_id: int
    is_leaf: bool = False
    
    # Split Info
    feature_indices: List[int]  # Which features used for split
    threshold: float            # Split threshold value
    
    # Children
    left: 'RLTNode'            # Left child (feature ≤ threshold)
    right: 'RLTNode'           # Right child (feature > threshold)
    
    # Stats
    prediction: float          # Leaf prediction value
    n_samples: int            # Number of samples in this node
    
    # Visualization Data
    muted_vars: List[int]                    # Variables excluded from this subtree
    pilot_importances: Dict[str, float]      # Feature importance scores from pilot
```

**Purpose**: Represents a single node in the RLT tree, storing all information needed for both prediction and visualization.

---

### 2. RLTBuilder - The Core Engine

This class builds individual trees using the RLT algorithm.

#### Key Methods:

**`__init__()`**: Configures the tree builder
- `reinforcement`: Enable/disable pilot models
- `muting_rate`: Percentage of variables to mute (e.g., 0.5 = mute bottom 50%)
- `nmin`: Minimum samples required to split a node
- `embed_config`: Configuration for pilot Random Forest models
- `record_history`: Whether to log the building process

**`fit()`**: Entry point for building a tree
```python
def fit(self, X, y, sample_indices):
    # Start recursive building from root
    root = self._fit_recursive(X, y, sample_indices, muted_set=set(), depth=0)
    return root, self.history_log
```

**`_fit_recursive()`**: The heart of the algorithm
```python
def _fit_recursive(self, X, y, current_indices, muted_set, depth, parent_id):
    # 1. Initialize node
    # 2. Check stopping criteria (min samples, purity, max depth)
    # 3. Run pilot model if reinforcement enabled
    # 4. Calculate variable importances and mute low-scoring features
    # 5. Find best split among remaining features
    # 6. Recursively build left and right children
    # 7. Return node
```

**How Reinforcement Works**:
```python
if run_reinforcement:
    # Run small Random Forest on current data
    est = RandomForestRegressor(n_estimators=10, max_depth=3)
    est.fit(X_embed, y_curr)
    
    # Get feature importances
    imps = est.feature_importances_
    
    # Mute bottom X% of features
    sorted_vars = sorted(features, key=lambda k: importance[k], reverse=True)
    n_mute = int(len(allowed) * self.muting_rate)
    new_mutes = set(sorted_vars[-n_mute:])  # Bottom performers
    
    # Children won't use muted variables
    child_muted_set = muted_set.union(new_mutes)
```

**`_find_split_random()`**: Finds the best split point
- Tries random subset of candidate features
- Tests random threshold values for each feature
- Uses variance reduction for regression (or Gini for classification)
- Returns feature and threshold with best gain

**`_make_leaf()`**: Creates a leaf node
- Calculates prediction (mean for regression, mode for classification)
- Logs event to training history

---

### 3. RLT Wrapper - Sklearn Interface

```python
class RLT(BaseEstimator):
    def __init__(self, ntrees=10, reinforcement=False, muting=-1, nmin=5):
        # Creates ensemble of RLT trees (like Random Forest)
        
    def fit(self, X, y, record_first_tree=True):
        # Builds multiple trees with bootstrap samples
        # Only records history for first tree (for visualization)
```

**Key Feature**: Only the **first tree** logs its training history for visualization. This keeps the JSON files manageable while still showing the algorithm in action.

---

### 4. Exporters - To JSON

**`export_static_tree()`**: Converts RLTNode tree to React Flow format
```python
def export_static_tree(node, feature_names=None):
    # Walks tree and creates:
    # - nodes_list: All nodes with their data
    # - edges_list: Connections between nodes
    # Returns format compatible with React Flow library
```

Output format:
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
        "mutedVars": ["Feature_1", "Feature_9"]
      }
    }
  ],
  "edges": [...]
}
```

---

### 5. Main Execution

```python
if __name__ == "__main__":
    # 1. Generate synthetic data (300 samples, 15 features)
    X, y = make_regression(n_samples=300, n_features=15, n_informative=5)
    
    # 2. Train RLT model with reinforcement
    model = RLT(ntrees=5, reinforcement=True, nmin=10)
    model.fit(X, y, record_first_tree=True)
    
    # 3. Export first tree for static visualization
    static_json = export_static_tree(model.trees_[0], feature_names)
    
    # 4. Export training history for animated playback
    training_history = model.histories_[0]
    
    # 5. Save both JSON files
```

**Output Files**:
1. `tree_data.json`: Final tree structure (for RLTTree.jsx)
2. `training_history.json`: Step-by-step events (for TrainingPlayer.jsx)

---

## React Visualization

The React app provides two visualization modes:

### Components:

#### 1. **App.jsx** - Main Entry Point
```jsx
export default function App() {
  return (
    <div>
      {/* Choose one: */}
      {/* <RLTTree data={treeData} /> */}          {/* Static view */}
      <TrainingPlayer history={historyData} />     {/* Animated view */}
    </div>
  );
}
```

**Two Modes**:
- **Static Mode** (`RLTTree`): Shows the final tree structure
- **Player Mode** (`TrainingPlayer`): Animates the building process step-by-step

---

#### 2. **RLTNode.jsx** - Custom Node Component

Displays individual tree nodes with:
- **Header**: Node type (Split/Leaf), sample count
- **Main Label**: 
  - Split nodes: "Feature_X ≤ threshold"
  - Leaf nodes: "Prediction: value"
- **Pilot Importance Chart**: Horizontal bar chart showing top 5 features
- **Muted Variables**: Red badges showing excluded features

**Key Sections**:
```jsx
{/* Pilot Importance Visualization */}
{data.pilotData && data.pilotData.length > 0 && (
  <BarChart width={250} height={110} data={data.pilotData} layout="vertical">
    <Bar dataKey="value" barSize={12} fill="#6366f1" />
  </BarChart>
)}

{/* Muted Variables */}
{data.mutedVars && data.mutedVars.length > 0 && (
  <div>
    {data.mutedVars.slice(0, 3).map(v => (
      <span>{v}</span>
    ))}
  </div>
)}
```

---

#### 3. **RLTTree.jsx** - Static Tree Viewer

Shows the complete tree at once using React Flow.

**How it works**:
1. Loads `tree_data.json`
2. Uses **Dagre** library to automatically layout nodes top-to-bottom
3. Renders using **React Flow** with:
   - Pan and zoom controls
   - Minimap for navigation
   - Custom RLTNode components

```jsx
const getLayoutedElements = (nodes, edges) => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setGraph({ rankdir: 'TB' }); // Top to Bottom
  
  // Add nodes and edges to Dagre
  // Calculate positions
  dagre.layout(dagreGraph);
  
  // Apply positions to React Flow nodes
  return layoutedNodes;
};
```

---

#### 4. **TrainingPlayer.jsx** - Animated Training Viewer

The most complex component - shows tree building step-by-step.

**Features**:
- ▶️ Play/Pause button
- Slider to scrub through steps
- Info box showing current event details
- Smooth camera follow
- Animated node appearance

**How Animation Works**:
```jsx
const renderStep = (stepIndex) => {
  // Replay all events up to current step
  for (let i = 0; i <= stepIndex; i++) {
    const event = history[i];
    
    if (event.type === 'NODE_INIT') {
      // Add new node to tree
    }
    else if (event.type === 'PILOT_RUN') {
      // Update node with pilot importance data
      // Highlight with purple border
    }
    else if (event.type === 'MUTE_VARS') {
      // Add muted variables to node
    }
    else if (event.type === 'SPLIT_DECISION') {
      // Update node with split info
    }
    else if (event.type === 'MAKE_LEAF') {
      // Convert to leaf node
    }
  }
  
  // Re-layout tree and smooth camera pan
  fitView({ duration: 800 });
};
```

**Event Types**:
1. **NODE_INIT**: New node created
2. **PILOT_RUN**: Random Forest runs to calculate importances
3. **MUTE_VARS**: Low-importance features excluded
4. **SPLIT_DECISION**: Best split point found
5. **MAKE_LEAF**: Node becomes a leaf (stopping criteria met)

---

## How It Works

### Complete Workflow:

```
1. DATA PREPARATION
   └─> Generate or load dataset (X, y)

2. PYTHON: BUILD RLT TREE
   ├─> Start at root with all samples
   ├─> For each node:
   │   ├─> Check stopping criteria
   │   ├─> Run pilot Random Forest
   │   ├─> Calculate feature importances
   │   ├─> Mute bottom X% of features
   │   ├─> Find best split among remaining features
   │   └─> Recursively build left and right children
   └─> Log each step to history

3. PYTHON: EXPORT TO JSON
   ├─> tree_data.json (final structure)
   └─> training_history.json (build steps)

4. REACT: LOAD & VISUALIZE
   ├─> Option A: Static view (show final tree)
   └─> Option B: Player view (animate construction)
```

---

### Example: Building a Node

**Step-by-step for a single node**:

1. **Initialize** (NODE_INIT)
   ```
   Node 0: 300 samples, depth 0
   Muted: []
   ```

2. **Run Pilot** (PILOT_RUN)
   ```
   Random Forest with 10 trees, max_depth=3
   Results:
     Feature_3: 0.35
     Feature_7: 0.22
     Feature_12: 0.18
     Feature_5: 0.12
     Feature_1: 0.08  ← Low importance
     ...
   ```

3. **Mute Variables** (MUTE_VARS)
   ```
   Muting bottom 50%:
   Muted: [Feature_1, Feature_4, Feature_6, Feature_9, ...]
   
   Remaining for split:
   Candidates: [Feature_3, Feature_7, Feature_12, Feature_5, ...]
   ```

4. **Find Split** (SPLIT_DECISION)
   ```
   Best split found:
   Feature_3 ≤ 0.52
   
   Left:  120 samples (Feature_3 ≤ 0.52)
   Right: 180 samples (Feature_3 > 0.52)
   ```

5. **Recurse**
   ```
   Build node.left  with muted_set=[Feature_1, Feature_4, ...]
   Build node.right with muted_set=[Feature_1, Feature_4, ...]
   ```

---

## Setup & Usage

### Python Side:

```bash
# Install dependencies
pip install numpy scikit-learn

# Run the script
python RLT_for_visualization.py

# Output:
# - tree_data.json
# - training_history.json
```

**Move JSON files** to `my-react-app/src/`

---

### React Side:

```bash
cd my-react-app

# Install dependencies
npm install

# Start dev server
npm run dev

# Open browser to http://localhost:5173
```

**Switch between modes** in `src/App.jsx`:
```jsx
// Static tree
import RLTTree from './RLTTree';
<RLTTree data={treeData} />

// Animated player
import TrainingPlayer from './TrainingPlayer';
<TrainingPlayer history={historyData} />
```

---

## Key Technologies

### Python:
- **NumPy**: Numerical computations
- **Scikit-learn**: Random Forest, base estimators, validation
- **Dataclasses**: Clean data structures

### React:
- **React Flow**: Graph visualization library
- **Dagre**: Automatic tree layout algorithm
- **Recharts**: Bar charts for pilot importances
- **Vite**: Fast development server

---

## Understanding the Algorithm Visually

### What You'll See:

1. **Node Creation** (Gray badge)
   - New node appears in tree
   - Shows sample count

2. **Pilot Run** (Purple badge)
   - Node highlights with purple border
   - Bar chart fills with importance scores
   - Shows top 5 most important features

3. **Variable Muting** (Red badge)
   - Red badges appear showing muted features
   - These won't be used in child nodes

4. **Split Decision** (Blue badge)
   - Node shows final split: "Feature_X ≤ threshold"
   - Purple border fades

5. **Leaf Creation** (Green badge)
   - Node turns into leaf
   - Shows prediction value
   - Background changes to green tint

---

## Configuration Options

### Python RLT Parameters:

```python
model = RLT(
    ntrees=10,           # Number of trees in ensemble
    reinforcement=True,  # Enable pilot models
    muting=0.5,         # Mute bottom 50% of features
    nmin=10,            # Minimum samples to split
    embed_config={
        'ntrees': 10,    # Pilot Random Forest trees
        'n_th': 10       # Min samples to run pilot
    }
)
```

### React Visualization:

In `TrainingPlayer.jsx`:
```javascript
interval = setInterval(() => {
  setCurrentStep(prev => prev + 1);
}, 600);  // Step duration (ms)
```

---

## Troubleshooting

**Issue**: JSON files not loading in React
- **Fix**: Ensure `tree_data.json` and `training_history.json` are in `my-react-app/src/`

**Issue**: Empty visualization
- **Fix**: Check browser console for errors
- **Fix**: Verify JSON files are valid (not empty)

**Issue**: Nodes overlapping
- **Fix**: Increase `nodeHeight` in layout calculation (RLTTree.jsx line 27)

**Issue**: Animation too fast/slow
- **Fix**: Adjust interval duration in TrainingPlayer.jsx line 276

---

## Advanced Topics

### Custom Datasets:

```python
# Load your own data
X = your_features  # Shape: (n_samples, n_features)
y = your_targets   # Shape: (n_samples,)
feature_names = ["Age", "Income", "Score", ...]

# Train
model = RLT(ntrees=5, reinforcement=True)
model.fit(X, y)

# Export with custom names
static_json = export_static_tree(model.trees_[0], feature_names)
```

### Experimenting with Muting Rates:

```python
# No muting (standard Random Forest behavior)
model = RLT(muting=0.0, reinforcement=True)

# Aggressive muting (only keep top 20%)
model = RLT(muting=0.8, reinforcement=True)

# Auto-calculate based on data
model = RLT(muting=-1, reinforcement=True)
```

---

## Research Context

This implementation is based on research into Reinforcement Learning Trees:
- Uses embedded models to guide tree growth
- Reduces feature space adaptively
- Balances accuracy and interpretability

See `RLT_notebook/` for experimental results comparing RLT to baseline models.

---

## License & Credits

Developed by Mohamed Abdennadher

Built with:
- Python (NumPy, Scikit-learn)
- React (React Flow, Recharts)
- Vite

---

## Summary

**RLT-prediction-visualization** demonstrates an advanced machine learning algorithm through interactive visualization:

1. **Python backend** implements the RLT algorithm with training history logging
2. **React frontend** visualizes both the final tree and the step-by-step building process
3. **Two viewing modes**: Static tree view or animated training player
4. **Educational tool**: Shows how reinforcement learning improves traditional decision trees

**Key Innovation**: Using pilot Random Forests to intelligently exclude noisy features at each node, leading to better generalization and more interpretable trees.
