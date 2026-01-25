import numpy as np
import pandas as pd
import json
import warnings
import os
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional

# Sklearn imports
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y
from joblib import Parallel, delayed  # Required for parallelization

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
#  1. DATA STRUCTURES
# ==============================================================================

@dataclass
class RLTNode:
    node_id: int
    is_leaf: bool = False
    
    # Splitting info
    feature_indices: List[int] = field(default_factory=list) 
    split_weights: List[float] = field(default_factory=list)
    threshold: float = None
    
    # Tree structure
    left: 'RLTNode' = None
    right: 'RLTNode' = None
    
    # Leaf content
    prediction: float = None
    model: object = None      
    
    # Metadata
    muted_vars: Set[int] = field(default_factory=set)
    sample_indices: np.ndarray = None
    
    # Visualization Data
    n_samples: int = 0
    pilot_importances: Dict[str, float] = field(default_factory=dict)

# ==============================================================================
#  2. INSTRUMENTED BUILDER
# ==============================================================================

class InstrumentedRLTBuilder:
    def __init__(self, model_type, nmin, mtry, alpha,
                 nsplit, split_gen,
                 reinforcement, muting_rate, protect_n,
                 combsplit, combsplit_th,
                 embed_config,       
                 track_obs, 
                 pilot_model,
                 record_history=False,
                 feature_names=None): # Added feature_names
        
        self.model_type = model_type
        self.nmin = nmin
        self.mtry = mtry
        self.alpha = alpha
        self.nsplit = nsplit
        self.split_gen = split_gen
        
        self.reinforcement = reinforcement
        self.muting_rate = muting_rate
        self.protect_n = protect_n
        self.embed_config = embed_config
        
        self.combsplit = combsplit
        self.combsplit_th = combsplit_th
        self.track_obs = track_obs
        self.pilot_model = pilot_model
        
        self.node_count = 0
        self.record_history = record_history
        self.history_log = []
        self.feature_names = feature_names # Store feature names

    def _get_feat_name(self, idx):
        """Helper to safely get feature name or fallback to Var X"""
        if self.feature_names and 0 <= idx < len(self.feature_names):
            return self.feature_names[idx]
        return f"Var {idx}"

    def fit(self, X, y, sample_indices_in_bag):
        self.history_log = []
        self.node_count = 0
        # Start recursion
        root = self._fit_recursive(X, y, sample_indices_in_bag, muted_set=set(), depth=0, parent_id=None)
        return root, self.history_log

    def _fit_recursive(self, X, y, current_indices, muted_set, depth, parent_id=None):
        my_id = self.node_count
        self.node_count += 1
        
        X_curr = X[current_indices]
        y_curr = y[current_indices]
        n_samples = len(current_indices)

        # 1. Log Node Initialization
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
        
        if n_samples < self.nmin or is_pure or depth > 20:
            return self._make_leaf(X_curr, y_curr, current_indices, muted_set, my_id, n_samples)

        # 3. Reinforcement (Pilot)
        run_reinforcement = self.reinforcement and (n_samples > self.embed_config['n_th'])
        
        captured_pilot = {}
        child_muted_set = muted_set
        candidate_pool = [i for i in range(X.shape[1]) if i not in muted_set]

        if run_reinforcement:
            allowed = [i for i in range(X.shape[1]) if i not in muted_set]
            if len(allowed) < 2: allowed = list(range(X.shape[1]))
            
            X_embed = X_curr[:, allowed]
            ntrees = self.embed_config['ntrees']
            embed_est = None
            raw_imps = None
            
            # Pilot Selection
            if self.pilot_model == "linear":
                scaler = StandardScaler()
                X_sc = scaler.fit_transform(X_embed)
                embed_est = Lasso(alpha=0.1) if self.model_type == "regression" else LogisticRegression(penalty='l1', solver='liblinear', C=0.5)
                embed_est.fit(X_sc, y_curr)
                if self.model_type == "regression":
                    raw_imps = np.abs(embed_est.coef_)
                else:
                    raw_imps = np.abs(embed_est.coef_).flatten()
            else:
                # Default to Random Forest
                if self.model_type == "regression":
                    embed_est = RandomForestRegressor(n_estimators=ntrees, max_depth=3, max_features='sqrt', n_jobs=1)
                else:
                    embed_est = RandomForestClassifier(n_estimators=ntrees, max_depth=3, max_features='sqrt', n_jobs=1)
                embed_est.fit(X_embed, y_curr)
                raw_imps = embed_est.feature_importances_

            # Capture Pilot Data
            if raw_imps is not None:
                VI = {str(allowed[i]): float(imp) for i, imp in enumerate(raw_imps)}
                captured_pilot = VI
                
                # Log Pilot with Real Names
                if self.record_history:
                    # Map indices to names for the JSON log
                    top_pilot = []
                    for k, v in sorted(VI.items(), key=lambda item: item[1], reverse=True)[:5]:
                        name = self._get_feat_name(int(k))
                        top_pilot.append({"name": name, "value": v})

                    self.history_log.append({
                        "step": len(self.history_log),
                        "type": "PILOT_RUN",
                        "nodeId": str(my_id),
                        "pilotData": top_pilot
                    })

            # Muting Logic
            if raw_imps is None or np.sum(raw_imps) == 0:
                child_muted_set = muted_set
                candidate_pool = allowed
            else:
                sorted_vars = sorted([int(k) for k in VI.keys()], key=lambda k: VI[str(k)], reverse=True)
                n_mute = min(max(0, len(allowed) - self.protect_n), int(len(allowed) * self.muting_rate))
                
                newly_muted = set(sorted_vars[-n_mute:]) if n_mute > 0 else set()
                child_muted_set = muted_set.union(newly_muted)
                
                # Candidates are sorted by importance, excluding muted
                candidate_pool = [v for v in sorted_vars if v not in newly_muted]

                # Log Muting with Real Names
                if self.record_history and newly_muted:
                    muted_names = [self._get_feat_name(m) for m in newly_muted]
                    self.history_log.append({
                        "step": len(self.history_log),
                        "type": "MUTE_VARS",
                        "nodeId": str(my_id),
                        "mutedVars": muted_names
                    })
                
        else:
            child_muted_set = muted_set
            candidate_pool = [i for i in range(X.shape[1]) if i not in muted_set]
            if not candidate_pool: candidate_pool = list(range(X.shape[1]))

        # 4. Splitting
        n_try = min(self.mtry, len(candidate_pool))
        split_candidates = candidate_pool[:n_try] if run_reinforcement else np.random.choice(candidate_pool, n_try, replace=False)
        
        best_split = self._find_best_split(X_curr, y_curr, split_candidates)

        if best_split is None:
            return self._make_leaf(X_curr, y_curr, current_indices, muted_set, my_id, n_samples)

        feat_idxs, weights, thr, left_mask, _ = best_split
        
        # Log Split Decision with Real Names
        if self.record_history:
            if len(feat_idxs) > 1:
                # Combination split
                names = [self._get_feat_name(i) for i in feat_idxs]
                feat_label = f"Comb({','.join(names)})"
            else:
                feat_label = self._get_feat_name(feat_idxs[0])
                
            self.history_log.append({
                "step": len(self.history_log),
                "type": "SPLIT_DECISION",
                "nodeId": str(my_id),
                "label": f"Thr: {thr:.2f}",
                "splitFeature": feat_label
            })

        node = RLTNode(node_id=my_id)
        node.feature_indices = feat_idxs
        node.split_weights = weights
        node.threshold = thr
        node.muted_vars = muted_set
        node.n_samples = n_samples
        node.pilot_importances = captured_pilot
        
        node.left = self._fit_recursive(X, y, current_indices[left_mask], child_muted_set, depth+1, my_id)
        node.right = self._fit_recursive(X, y, current_indices[~left_mask], child_muted_set, depth+1, my_id)
        return node
    
    def _find_best_split(self, X, y, candidates):
        best_gain = -np.inf
        best_cfg = None
        iterations = max(self.nsplit, 10) if (self.combsplit > 1 and len(candidates) >= self.combsplit) else 1
        
        for _ in range(iterations):
            if self.combsplit > 1 and len(candidates) >= self.combsplit:
                c_vars = np.random.choice(candidates, self.combsplit, replace=False)
                c_weights = np.random.choice([-1, 1], size=self.combsplit)
                vals = np.dot(X[:, c_vars], c_weights)
                res = self._optimize_threshold(vals, y)
                if res and res[2] > best_gain:
                    best_gain = res[2]
                    best_cfg = (c_vars.tolist(), c_weights.tolist(), res[0], res[1], res[2])
            else:
                for f in candidates:
                    vals = X[:, f]
                    res = self._optimize_threshold(vals, y)
                    if res and res[2] > best_gain:
                        best_gain = res[2]
                        best_cfg = ([f], [1.0], res[0], res[1], res[2])
        return best_cfg

    def _optimize_threshold(self, values, y):
        unique_vals = np.unique(values)
        if len(unique_vals) < 2: return None
        thresholds = unique_vals
        if self.nsplit > 0 and self.split_gen == "random":
             thresholds = np.random.choice(unique_vals, size=min(self.nsplit, len(unique_vals)), replace=False)
        best = -np.inf
        best_t, best_m = None, None
        n = len(y)
        for t in thresholds:
            l_mask = values <= t
            n_l = np.sum(l_mask)
            if n_l < self.alpha * n or (n-n_l) < self.alpha * n: continue
            if n_l < self.nmin or (n-n_l) < self.nmin: continue
            if self.model_type == "regression":
                gain = np.var(y)*n - (np.var(y[l_mask])*n_l + np.var(y[~l_mask])*(n-n_l))
            else:
                def g(z): return 1.0 - np.sum((np.unique(z, return_counts=True)[1]/len(z))**2)
                gain = g(y) - ((n_l/n)*g(y[l_mask]) + ((n-n_l)/n)*g(y[~l_mask]))
            if gain > best: best, best_t, best_m = gain, t, l_mask
        return (best_t, best_m, best) if best_t is not None else None

    def _make_leaf(self, X_leaf, y_leaf, current_indices, muted_set, my_id, n_samples):
        node = RLTNode(node_id=my_id, is_leaf=True)
        node.n_samples = n_samples
        node.muted_vars = muted_set
        if self.track_obs: node.sample_indices = current_indices

        if self.model_type == "regression":
            node.prediction = np.mean(y_leaf)
        else:
            vals, counts = np.unique(y_leaf, return_counts=True)
            node.prediction = vals[np.argmax(counts)]

        if self.record_history:
            self.history_log.append({
                "step": len(self.history_log),
                "type": "MAKE_LEAF",
                "nodeId": str(my_id),
                "label": f"Pred: {node.prediction:.2f}"
            })

        # Fit Linear Model (Ridge/Logistic)
        if len(y_leaf) >= 5 and len(np.unique(y_leaf)) >= 2:
            try:
                if self.model_type == "regression":
                    if np.std(y_leaf) > 1e-9:
                        model = Ridge(alpha=1.0)
                        model.fit(X_leaf, y_leaf)
                        node.model = model
                else:
                    if len(np.unique(y_leaf)) > 1:
                        model = LogisticRegression(max_iter=100, solver='liblinear')
                        model.fit(X_leaf, y_leaf)
                        node.model = model
            except Exception:
                node.model = None
        return node

# ==============================================================================
#  3. INSTRUMENTED RLT ESTIMATOR
# ==============================================================================

class InstrumentedRLT(BaseEstimator): 
    def __init__(self, 
                 ntrees=100, 
                 model="regression",
                 pilot_model="random_forest", 
                 mtry=None, 
                 nmin=None, 
                 alpha=0.1, 
                 split_gen="random", 
                 nsplit=1,
                 resample_prob=0.9, 
                 replacement=True,
                 reinforcement=True,
                 muting=-1,
                 protect=None,
                 combsplit=1,
                 combsplit_th=0.25,
                 track_obs=False,
                 embed_ntrees=50,
                 embed_mtry=0.5,
                 embed_nmin=None,
                 embed_n_th=None,
                 n_jobs=-1):
        
        self.ntrees = ntrees
        self.model = model
        self.pilot_model = pilot_model
        self.mtry = mtry
        self.nmin = nmin
        self.alpha = alpha
        self.split_gen = split_gen
        self.nsplit = nsplit
        self.resample_prob = resample_prob
        self.replacement = replacement
        self.reinforcement = reinforcement
        self.muting = muting
        self.protect = protect
        self.combsplit = combsplit
        self.combsplit_th = combsplit_th
        self.track_obs = track_obs
        self.embed_ntrees = embed_ntrees
        self.embed_mtry = embed_mtry
        self.embed_nmin = embed_nmin
        self.embed_n_th = embed_n_th
        self.n_jobs = n_jobs
        
        self.trees_ = []
        self.histories_ = [] 

    def fit(self, X, y, feature_names=None, record_first_tree=True):
        """
        Modified fit method to accept feature_names and use Parallel execution.
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        
        if self.mtry is None: self.mtry = max(1, int(n_features / 3))
        if self.nmin is None: self.nmin = max(1, int(np.log(n_samples)))
        if self.protect is None: self.protect = int(np.log(n_features))
        if self.embed_nmin is None: self.embed_nmin = int(n_samples**(1/3))
        if self.embed_n_th is None: self.embed_n_th = 4 * self.nmin
        
        if self.muting == -1:
            if self.reinforcement:
                base_term = np.log(n_features) / n_features if n_features > 1 else 0
                offset = 0.35 
                real_muting = max(0.0, min(0.9, base_term + offset))
            else:
                real_muting = 0.0
        else:
            real_muting = self.muting

        embed_config = {
            'ntrees': self.embed_ntrees,
            'mtry': max(1, int(self.embed_mtry * n_features)) if isinstance(self.embed_mtry, float) else self.embed_mtry,
            'nmin': self.embed_nmin,
            'n_th': self.embed_n_th
        }
        
        seeds = np.random.randint(0, 1e9, self.ntrees)
        
        # --- PARALLEL EXECUTION START ---
        # We process trees in parallel using joblib.
        # Only the first tree (index 0) gets record_history=True.
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_tree_instrumented)(
                X, y, seeds[i], real_muting, embed_config, 
                record_history=(i == 0 and record_first_tree),
                feature_names=feature_names
            )
            for i in range(self.ntrees)
        )
        
        # Unzip the results: results is a list of tuples (tree, history)
        self.trees_ = [res[0] for res in results]
        
        # Extract histories (only keeping non-empty ones, usually just the first)
        self.histories_ = [res[1] for res in results if res[1]]
        
        return self

    def _fit_single_tree_instrumented(self, X, y, seed, muting_rate, embed_config, record_history, feature_names):
        np.random.seed(seed)
        n_samples = X.shape[0]
        sample_size = int(n_samples * self.resample_prob)
        indices = np.random.choice(n_samples, sample_size, replace=self.replacement)
        
        builder = InstrumentedRLTBuilder(
            model_type=self.model,
            nmin=self.nmin,
            mtry=self.mtry,
            alpha=self.alpha,
            nsplit=self.nsplit,
            split_gen=self.split_gen,
            reinforcement=self.reinforcement,
            muting_rate=muting_rate,
            protect_n=self.protect,
            combsplit=self.combsplit,
            combsplit_th=self.combsplit_th,
            embed_config=embed_config,
            track_obs=self.track_obs,
            pilot_model=self.pilot_model,
            record_history=record_history,
            feature_names=feature_names # Pass features to builder
        )
        return builder.fit(X, y, indices)

# ==============================================================================
#  4. EXPORTERS (Logic to create valid JSON)
# ==============================================================================

def export_static_tree(node, feature_names=None):
    nodes_list = []
    edges_list = []
    
    def walk(curr, parent_id=None, edge_label=None):
        pilot_data = []
        if curr.pilot_importances:
            sorted_items = sorted(curr.pilot_importances.items(), key=lambda x: x[1], reverse=True)
            for k, v in sorted_items[:5]:
                name = feature_names[int(k)] if feature_names and k.isdigit() else f"Var {k}"
                pilot_data.append({"name": name, "value": v})
        
        muted = []
        for m in curr.muted_vars:
            muted.append(feature_names[m] if feature_names else f"Var {m}")
            
        feat_name = ""
        node_label = ""
        
        if curr.is_leaf:
            node_label = f"Pred: {curr.prediction:.2f}"
            if curr.model:
                node_label += f"\n({curr.model.__class__.__name__})"
        else:
            if len(curr.feature_indices) == 1:
                idx = curr.feature_indices[0]
                feat_name = feature_names[idx] if feature_names else f"Var {idx}"
            else:
                feat_name = "Comb(" + ",".join([str(i) for i in curr.feature_indices]) + ")"
            node_label = f"Thr: {curr.threshold:.2f}"

        nodes_list.append({
            "id": str(curr.node_id),
            "type": "rltNode",
            "data": {
                "isLeaf": curr.is_leaf,
                "label": node_label,
                "splitFeature": feat_name,
                "samples": int(curr.n_samples),
                "pilotData": pilot_data,
                "mutedVars": muted
            },
            "position": {"x": 0, "y": 0} 
        })
        
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
    print("1. 🎲 Loading Data...")
    
    dataset_names_reg = ["AutoMPG", "crime", "concrete", "energy_cooling_load", "energy_heating_load"]  
    
    # Ensure output directory exists
    os.makedirs("rlt_output", exist_ok=True)
    
    for dataset in dataset_names_reg:
        print(f"\n--- Processing: {dataset} ---")
        
        try:
            # ---------------------------
            # Load dataset
            # ---------------------------
            # Adjust path as needed
            x_path = f"RLT_notebook/datasets/regression/{dataset}/X_train_data.csv"
            y_path = f"RLT_notebook/datasets/regression/{dataset}/y_train_data.csv"
            
            if not os.path.exists(x_path):
                print(f"⚠️ Warning: File not found {x_path}. Skipping.")
                continue

            # Load DataFrames
            df_X = pd.read_csv(x_path)
            df_y = pd.read_csv(y_path)
            
            # Extract Feature Names (Crucial for visualization labels)
            feature_names = df_X.columns.tolist()
            
            # Convert to Numpy for Training (Uppercase X)
            X = df_X.to_numpy()
            y = df_y.to_numpy().ravel()
            
            print(f"   Loaded {X.shape[0]} samples, {X.shape[1]} features.")
        
            print("2. 🚀 Training RLT Model (with Recording)...")
            model = InstrumentedRLT(
                    n_jobs=-1,  # Uses all available cores
                    model="regression",
                    split_gen ="random",
                    resample_prob=0.9,
                    reinforcement = True,
                    protect=2,
                    pilot_model = "random_forest",
                    nsplit = 1,
                    nmin=20,
                    muting=-1,
                    mtry=5,
                    combsplit=1,
                    alpha=0.05,
            )
            
            # PASS feature_names HERE
            model.fit(X, y, feature_names=feature_names, record_first_tree=True)
            
            first_tree = model.trees_[0]
            training_history = model.histories_[0]
            
            # ---------------------------
            # Save JSONs
            # ---------------------------
            static_filename = f"tree_data_{dataset}.json"
            history_filename = f"training_history_{dataset}.json"
            
            print(f"3. 💾 Saving '{static_filename}' (Static View)...")
            static_json = export_static_tree(first_tree, feature_names)
            with open(static_filename, "w") as f:
                json.dump(static_json, f, indent=2)
                
            print(f"4. 💾 Saving '{history_filename}' (Player View)...")
            with open(history_filename, "w") as f:
                json.dump(training_history, f, indent=2)
                
            print(f"✅ Success for {dataset}!")
            
        except Exception as e:
            print(f"❌ Error processing {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n🎉 All datasets processed.")