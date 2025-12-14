import numpy as np
import json
import warnings
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional

# Sklearn imports
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.utils.validation import check_X_y, check_array
from sklearn.datasets import make_regression

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
#  1. DATA STRUCTURES
# ==============================================================================

@dataclass
class RLTNode:
    node_id: int
    is_leaf: bool = False
    
    # Split Info
    feature_indices: List[int] = field(default_factory=list) 
    threshold: float = None
    
    # Children
    left: 'RLTNode' = None
    right: 'RLTNode' = None
    
    # Stats
    prediction: float = None
    n_samples: int = 0
    
    # Visualization Data
    muted_vars: List[int] = field(default_factory=list)
    pilot_importances: Dict[str, float] = field(default_factory=dict)

# ==============================================================================
#  2. RLT BUILDER (The Core Engine)
# ==============================================================================

class RLTBuilder:
    def __init__(self, model_type, nmin, mtry, alpha,
                nsplit, split_gen,
                reinforcement, muting_rate, protect_n,
                embed_config,       
                leaf_embed_config,  
                record_history=False): # NEW: Flag to enable/disable history logging
        
        self.model_type = model_type
        self.nmin = nmin
        self.mtry = mtry
        self.alpha = alpha
        self.nsplit = nsplit
        self.split_gen = split_gen
        
        self.reinforcement = reinforcement
        self.muting_rate = muting_rate
        self.protect_n = protect_n
        
        # --- SAFE CONFIG LOADING (Fixes KeyErrors/NoneType) ---
        self.embed_config = embed_config if embed_config is not None else {}
        self.leaf_embed_config = leaf_embed_config if leaf_embed_config is not None else {}
        
        self.record_history = record_history
        self.history_log = [] # Stores step-by-step events
        self.node_count = 0

    def fit(self, X, y, sample_indices):
        self.history_log = [] # Reset history
        self.node_count = 0
        root = self._fit_recursive(X, y, sample_indices, muted_set=set(), depth=0, parent_id=None)
        return root, self.history_log

    def _fit_recursive(self, X, y, current_indices, muted_set, depth, parent_id):
        # 1. Setup Node
        my_id = self.node_count
        self.node_count += 1
        
        X_curr = X[current_indices]
        y_curr = y[current_indices]
        n_samples = len(current_indices)

        # --- LOG HISTORY: NODE INIT ---
        if self.record_history:
            self.history_log.append({
                "step": len(self.history_log),
                "type": "NODE_INIT",
                "nodeId": str(my_id),
                "parentId": str(parent_id) if parent_id is not None else None,
                "depth": depth,
                "samples": int(n_samples)
            })

        # 2. Check Stopping Criteria
        is_pure = (len(np.unique(y_curr)) == 1) if self.model_type == "classification" else (np.var(y_curr) < 1e-8)
        
        if n_samples < self.nmin or is_pure or depth > 10: # Limit depth for demo
            return self._make_leaf(y_curr, my_id, n_samples, muted_set)

        # 3. Reinforcement Logic (Pilot Model)
        # --- SAFE LOOKUP: Use .get() ---
        n_th = self.embed_config.get('n_th', 10)
        run_reinforcement = self.reinforcement and (n_samples > n_th)
        
        captured_pilot = {}
        child_muted_set = muted_set
        candidate_pool = [i for i in range(X.shape[1]) if i not in muted_set]

        if run_reinforcement:
            allowed = [i for i in range(X.shape[1]) if i not in muted_set]
            if len(allowed) < 2: allowed = list(range(X.shape[1]))
            
            # Run Pilot
            X_embed = X_curr[:, allowed]
            ntrees_pilot = self.embed_config.get('ntrees', 10) # Default 10 if missing
            
            est = RandomForestRegressor(n_estimators=ntrees_pilot, max_depth=3, n_jobs=1)
            est.fit(X_embed, y_curr)
            
            # Capture Importances
            imps = est.feature_importances_
            VI = {str(allowed[i]): float(imp) for i, imp in enumerate(imps)}
            captured_pilot = VI

            # --- LOG HISTORY: PILOT RUN ---
            if self.record_history:
                top_pilot = [{"name": f"Var {k}", "value": v} for k, v in sorted(VI.items(), key=lambda x: x[1], reverse=True)[:5]]
                self.history_log.append({
                    "step": len(self.history_log),
                    "type": "PILOT_RUN",
                    "nodeId": str(my_id),
                    "pilotData": top_pilot
                })

            # Calculate Muting
            sorted_vars = sorted([int(k) for k in VI.keys()], key=lambda k: VI[str(k)], reverse=True)
            n_mute = int(len(allowed) * self.muting_rate)
            new_mutes = set(sorted_vars[-n_mute:]) if n_mute > 0 else set()
            
            child_muted_set = muted_set.union(new_mutes)
            candidate_pool = [x for x in sorted_vars if x not in new_mutes]

            # --- LOG HISTORY: MUTING ---
            if self.record_history and new_mutes:
                self.history_log.append({
                    "step": len(self.history_log),
                    "type": "MUTE_VARS",
                    "nodeId": str(my_id),
                    "mutedVars": [f"Var {x}" for x in child_muted_set]
                })

        # 4. Find Best Split
        # Fallback if everything is muted
        if not candidate_pool: candidate_pool = [0]
        
        split_feat, split_val = self._find_split_random(X_curr, y_curr, candidate_pool)
        
        if split_feat is None:
             return self._make_leaf(y_curr, my_id, n_samples, muted_set)

        # 5. Create Split Node
        left_mask = X_curr[:, split_feat] <= split_val
        
        # --- LOG HISTORY: DECISION ---
        if self.record_history:
            self.history_log.append({
                "step": len(self.history_log),
                "type": "SPLIT_DECISION",
                "nodeId": str(my_id),
                "label": f"Threshold: {split_val:.2f}",
                "splitFeature": f"Var {split_feat}"
            })

        node = RLTNode(node_id=my_id)
        node.feature_indices = [int(split_feat)]
        node.threshold = float(split_val)
        node.n_samples = n_samples
        node.muted_vars = [int(x) for x in muted_set]
        node.pilot_importances = captured_pilot
        
        node.left = self._fit_recursive(X, y, current_indices[left_mask], child_muted_set, depth+1, my_id)
        node.right = self._fit_recursive(X, y, current_indices[~left_mask], child_muted_set, depth+1, my_id)
        
        return node

    def _find_split_random(self, X, y, candidates):
        """ Simplified random splitting for stability """
        best_gain = -np.inf
        best_f, best_t = None, None
        
        # Try 5 random features
        try_feats = np.random.choice(candidates, size=min(5, len(candidates)), replace=False)
        
        for f in try_feats:
            vals = X[:, f]
            # Try 5 random thresholds
            unique_vals = np.unique(vals)
            if len(unique_vals) < 2: continue
            
            thresholds = np.random.choice(unique_vals, size=min(5, len(unique_vals)), replace=False)
            
            for t in thresholds:
                left = vals <= t
                if np.sum(left) < self.nmin or np.sum(~left) < self.nmin: continue
                
                # Variance reduction (Regression)
                current_var = np.var(y)
                l_var = np.var(y[left]) if np.sum(left) > 0 else 0
                r_var = np.var(y[~left]) if np.sum(~left) > 0 else 0
                gain = current_var - ( (np.sum(left)/len(y))*l_var + (np.sum(~left)/len(y))*r_var )
                
                if gain > best_gain:
                    best_gain = gain
                    best_f, best_t = f, t
                    
        return best_f, best_t

    def _make_leaf(self, y_curr, node_id, n_samples, muted_set):
        pred = np.mean(y_curr) if len(y_curr) > 0 else 0.0
        
        if self.record_history:
            self.history_log.append({
                "step": len(self.history_log),
                "type": "MAKE_LEAF",
                "nodeId": str(node_id),
                "label": f"Pred: {pred:.2f}"
            })
            
        return RLTNode(
            node_id=node_id, 
            is_leaf=True, 
            prediction=pred, 
            n_samples=n_samples,
            muted_vars=[int(x) for x in muted_set]
        )

# ==============================================================================
#  3. RLT WRAPPER (Sklearn Interface)
# ==============================================================================

class RLT(BaseEstimator):
    def __init__(self, ntrees=10, reinforcement=False, muting=-1, nmin=5, embed_config=None):
        self.ntrees = ntrees
        self.reinforcement = reinforcement
        self.muting = muting
        self.nmin = nmin
        # Default Configs to ensure no NoneTypes
        self.embed_config = embed_config if embed_config else {'ntrees': 10, 'n_th': 10}
        self.trees_ = []
        self.histories_ = []

    def fit(self, X, y, record_first_tree=True):
        self.trees_ = []
        n_samples = X.shape[0]
        
        # Auto-calc muting if needed
        muting_rate = 0.5 if self.muting == -1 and self.reinforcement else max(0, self.muting)

        # Build Trees
        for i in range(self.ntrees):
            is_recording = (i == 0 and record_first_tree) # Only record history for the first tree
            
            builder = RLTBuilder(
                model_type="regression",
                nmin=self.nmin,
                mtry=max(1, int(X.shape[1]/3)),
                alpha=0.1, nsplit=1, split_gen="random",
                reinforcement=self.reinforcement,
                muting_rate=muting_rate,
                protect_n=2,
                embed_config=self.embed_config,
                leaf_embed_config={},
                record_history=is_recording
            )
            
            indices = np.random.choice(n_samples, n_samples, replace=True)
            root, history = builder.fit(X, y, indices)
            
            self.trees_.append(root)
            if is_recording:
                self.histories_.append(history)
                
        return self

# ==============================================================================
#  4. EXPORTERS (To JSON)
# ==============================================================================

def export_static_tree(node, feature_names=None):
    """ Exports the final structure for the Static Viewer """
    nodes_list = []
    edges_list = []
    
    def walk(curr, parent_id=None, edge_label=None):
        # Format Pilot Data
        pilot_data = []
        if curr.pilot_importances:
            sorted_items = sorted(curr.pilot_importances.items(), key=lambda x: x[1], reverse=True)
            for k, v in sorted_items[:5]:
                name = feature_names[int(k)] if feature_names and k.isdigit() else f"Var {k}"
                pilot_data.append({"name": name, "value": v})
        
        # Format Muted
        muted = []
        for m in curr.muted_vars:
            muted.append(feature_names[m] if feature_names else f"Var {m}")
            
        # Feature Name
        feat_name = ""
        if not curr.is_leaf and curr.feature_indices:
            idx = curr.feature_indices[0]
            feat_name = feature_names[idx] if feature_names else f"Var {idx}"

        # Node Object
        nodes_list.append({
            "id": str(curr.node_id),
            "type": "rltNode",
            "data": {
                "isLeaf": curr.is_leaf,
                "label": f"Pred: {curr.prediction:.2f}" if curr.is_leaf else f"Threshold: {curr.threshold:.2f}",
                "splitFeature": feat_name,
                "samples": curr.n_samples,
                "pilotData": pilot_data,
                "mutedVars": muted
            },
            "position": {"x": 0, "y": 0}
        })
        
        # Edge Object
        if parent_id is not None:
            edges_list.append({
                "id": f"e{parent_id}-{curr.node_id}",
                "source": str(parent_id),
                "target": str(curr.node_id),
                "label": edge_label,
                "type": "smoothstep",
                "animated": True
            })
            
        if curr.left: walk(curr.left, curr.node_id, "True")
        if curr.right: walk(curr.right, curr.node_id, "False")

    walk(node)
    return {"nodes": nodes_list, "edges": edges_list}

# ==============================================================================
#  5. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("1. 🎲 Generating Data...")
    X, y = make_regression(n_samples=300, n_features=15, n_informative=5, noise=0.1, random_state=42)
    feature_names = [f"Feature_{i}" for i in range(15)]
    
    print("2. 🚀 Training RLT Model (with Recording)...")
    # reinforcement=True enables the Pilot models
    model = RLT(ntrees=5, reinforcement=True, nmin=10)
    model.fit(X, y, record_first_tree=True)
    
    first_tree = model.trees_[0]
    training_history = model.histories_[0]
    
    print("3. 💾 Saving 'tree_data.json' (Static View)...")
    static_json = export_static_tree(first_tree, feature_names)
    with open("sec/tree_data.json", "w") as f:
        json.dump(static_json, f, indent=2)
        
    print("4. 💾 Saving 'src/training_history.json' (Player View)...")
    with open("training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
        
    print("\n✅ Success! Move both JSON files to your React 'src' folder.")