import React from 'react';
// Uncomment the one you want to use:
// import RLTTree from './RLTTree'; 
import TrainingPlayer from './TrainingPlayer'; 

// Load the exported JSON files from Python
import treeData from './tree_data.json';
import historyData from './training_history.json';

/**
 * Main App Component
 * 
 * Choose between two visualization modes:
 * 1. RLTTree: Shows the complete final tree (static)
 * 2. TrainingPlayer: Animates the tree building process step-by-step
 * 
 * To switch modes, uncomment/comment the imports and JSX below.
 */
export default function App() {
  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* 2. Graph Container (Fills rest of screen) */}
      <div style={{ flex: 1, position: 'relative' }}>
        
        {/* OPTION A: The Static Tree */}
        {/* Uncomment this to show the final tree all at once */}
        {/* <RLTTree data={treeData} /> */}

        {/* OPTION B: The Training Player */}
        {/* Shows animated step-by-step tree construction */}
        <TrainingPlayer history={historyData} />

      </div>
    </div>
  );
}