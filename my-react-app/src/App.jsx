import React from 'react';
// Uncomment the one you want to use:
// import RLTTree from './RLTTree'; 
import TrainingPlayer from './TrainingPlayer'; 

import treeData from './tree_data.json';
import historyData from './training_history.json';

export default function App() {
  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* 2. Graph Container (Fills rest of screen) */}
      <div style={{ flex: 1, position: 'relative' }}>
        
        {/* OPTION A: The Static Tree */}
        {/* <RLTTree data={treeData} /> */}

        {/* OPTION B: The Training Player (Uncomment this to use) */}
        <TrainingPlayer history={historyData} />

      </div>
    </div>
  );
}