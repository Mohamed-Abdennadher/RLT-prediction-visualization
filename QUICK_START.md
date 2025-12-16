# Quick Start Guide

## What is this project?

This project visualizes **Reinforcement Learning Trees (RLT)** - a machine learning algorithm that builds smarter decision trees by:
1. Running "pilot" models at each node to find important features
2. Muting (excluding) unimportant features from child nodes
3. Making more focused, accurate predictions

You get **two visualization modes**:
- **Static Tree**: See the final tree structure
- **Training Player**: Watch the tree build step-by-step (like a video)

---

## 5-Minute Setup

### Step 1: Run Python Script
```bash
# Generate the tree data
python RLT_for_visualization.py
```

**Output**: Two JSON files
- `tree_data.json` - Final tree structure
- `training_history.json` - Building process events

### Step 2: Move Files to React App
```bash
# Move generated files to React src folder
mv tree_data.json my-react-app/src/
mv training_history.json my-react-app/src/
```

### Step 3: Start React App
```bash
cd my-react-app
npm install   # First time only
npm run dev
```

Open browser to: **http://localhost:5173**

---

## What You'll See

### Training Player View (Default)

**Controls**:
- ▶️ **Play/Pause**: Auto-advance through steps
- **Slider**: Scrub to any point in the process
- **Info Panel**: Shows what's happening at each step

**Visual Events**:

1. **Gray Badge - NODE_INIT**
   - New node appears
   - Shows sample count

2. **Purple Badge - PILOT_RUN**
   - Node highlights purple
   - Bar chart shows feature importances
   - "Running embedded Random Forest..."

3. **Red Badge - MUTE_VARS**
   - Red badges appear for excluded features
   - "Muting X noisy variables..."

4. **Blue Badge - SPLIT_DECISION**
   - Shows final split: "Feature_X ≤ 0.52"
   - Purple highlight fades

5. **Green Badge - MAKE_LEAF**
   - Node becomes terminal
   - Shows prediction value
   - Background tints green

---

## Switching Visualization Modes

Edit `my-react-app/src/App.jsx`:

```jsx
// For STATIC TREE VIEW:
import RLTTree from './RLTTree';
<RLTTree data={treeData} />

// For ANIMATED TRAINING (default):
import TrainingPlayer from './TrainingPlayer';
<TrainingPlayer history={historyData} />
```

---

## Understanding the Algorithm

### Normal Decision Tree:
```
Node -> Check all features -> Find best split -> Repeat
```

### RLT Decision Tree:
```
Node -> Run pilot Random Forest
     -> Rank features by importance
     -> MUTE bottom 50% of features
     -> Find best split using top 50% only
     -> Repeat (with accumulated muted list)
```

**Why is this better?**
- Ignores noisy/irrelevant features automatically
- Reduces overfitting
- Creates more interpretable trees
- Each node focuses on what actually matters

---

## Customizing the Algorithm

### In Python (`RLT_for_visualization.py`):

```python
model = RLT(
    ntrees=10,           # Number of trees in ensemble
    reinforcement=True,  # Enable/disable pilot models
    muting=0.5,         # Mute bottom 50% (try 0.3 or 0.7)
    nmin=10,            # Min samples to split (try 5 or 20)
)
```

**Experiment Ideas**:
- `muting=0.0`: No muting (like regular Random Forest)
- `muting=0.8`: Aggressive muting (only top 20%)
- `reinforcement=False`: Disable pilots entirely

### Using Your Own Data:

```python
# Replace make_regression with your data
X = your_features  # Shape: (n_samples, n_features)
y = your_targets   # Shape: (n_samples,)
feature_names = ["Age", "Income", "Score", ...]

model = RLT(ntrees=5, reinforcement=True)
model.fit(X, y)

# Export with your feature names
static_json = export_static_tree(model.trees_[0], feature_names)
```

---

## Animation Speed

Too fast/slow? Edit `my-react-app/src/TrainingPlayer.jsx` line ~276:

```javascript
interval = setInterval(() => {
  setCurrentStep(prev => prev + 1);
}, 600);  // Change this number (milliseconds)
```

- `300` = Fast (0.3s per step)
- `600` = Default (0.6s per step)
- `1200` = Slow (1.2s per step)

---

## Key Files Explained

### Python:
- **`RLT_for_visualization.py`** - Complete RLT implementation
  - `RLTNode`: Tree node data structure
  - `RLTBuilder`: Core algorithm (the interesting part!)
  - `RLT`: Ensemble wrapper (builds multiple trees)
  - `export_static_tree()`: Converts to JSON

### React:
- **`App.jsx`** - Choose which view to show
- **`RLTNode.jsx`** - How each node looks (card with charts)
- **`RLTTree.jsx`** - Static tree viewer
- **`TrainingPlayer.jsx`** - Animated step-by-step viewer

### Data:
- **`tree_data.json`** - Final tree structure
- **`training_history.json`** - Event log for animation

---

## Troubleshooting

### "No data loaded"
- ✅ Run Python script first
- ✅ Move JSON files to `my-react-app/src/`
- ✅ Restart React dev server (`npm run dev`)

### Blank screen
- ✅ Check browser console (F12) for errors
- ✅ Verify JSON files aren't empty
- ✅ Try `npm install` again

### Nodes overlapping
- Edit `RLTTree.jsx` line 27: increase `nodeHeight`
- Edit `TrainingPlayer.jsx` line 155: increase height value

---

## Next Steps

1. **Read the full README.md** for deep technical details
2. **Experiment with parameters** (muting rate, min samples)
3. **Try your own dataset** (classification or regression)
4. **Check `RLT_notebook/`** for performance comparisons

---

## Quick Reference

| What do you want? | How to do it |
|-------------------|--------------|
| See final tree | Uncomment `<RLTTree>` in App.jsx |
| See animated build | Use `<TrainingPlayer>` (default) |
| Use my data | Edit Python script, replace `make_regression()` |
| Change muting % | Edit `muting=0.5` in Python |
| Slower animation | Increase interval in TrainingPlayer.jsx |
| Custom feature names | Pass list to `export_static_tree()` |
| More/fewer trees | Change `ntrees=` parameter |

---

## Learn More

- **Algorithm details**: See README.md "How It Works" section
- **Code walkthrough**: See README.md "Python Implementation" section
- **Research background**: Check `RLT_notebook/` folder

---

**Happy visualizing! 🎉**

For questions or issues, check the main README.md or open an issue on GitHub.
