import React, { useState, useEffect } from 'react';
import RLTTree from './RLTTree';

export default function ModelDashboard() {
const [treeData, setTreeData] = useState(null);

useEffect(() => {
// Replace with your actual Python endpoint
fetch('/api/get-rlt-structure')
    .then(res => res.json())
    .then(data => setTreeData(data));
}, []);

if (!treeData) return <div>Loading RLT Model...</div>;

return (
<div className="p-8 bg-white">
    <h1 className="text-2xl font-bold mb-4 text-gray-800">RLT Model Inspector</h1>
    <p className="mb-4 text-gray-500">Visualizing Reinforcement Learning steps, Embedded Pilots, and Muted Variables.</p>
    
    <RLTTree initialData={treeData} />
</div>
);
}