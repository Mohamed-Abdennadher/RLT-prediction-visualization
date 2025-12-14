/* eslint-disable react/prop-types */
import React, { useEffect } from 'react';
import ReactFlow, { 
  useNodesState, 
  useEdgesState, 
  Controls, 
  Background, 
  MiniMap 
} from 'reactflow';
import dagre from 'dagre';
import 'reactflow/dist/style.css'; 

import RLTNode from './RLTNode'; 

const nodeTypes = { rltNode: RLTNode };

// --- LAYOUT ALGORITHM (Safe Version) ---
const getLayoutedElements = (nodes, edges) => {
  // 1. Safety Check: If data is missing, stop immediately
  if (!nodes || nodes.length === 0) return { nodes: [], edges: [] };
  if (!edges) edges = [];

  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));

  const nodeWidth = 320;
  const nodeHeight = 300; // Increased height to fit charts

  dagreGraph.setGraph({ rankdir: 'TB' }); // TB = Top to Bottom

  // 2. Add Nodes to Dagre
  nodes.forEach((node) => {
    // Safety: Ensure ID exists
    if(node.id) {
        dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
    }
  });

  // 3. Add Edges to Dagre
  edges.forEach((edge) => {
    // Safety: Ensure source and target exist
    if(edge.source && edge.target) {
        dagreGraph.setEdge(edge.source, edge.target);
    }
  });

  // 4. Calculate Layout
  dagre.layout(dagreGraph);

  // 5. Apply positions back to React Flow nodes
  const layoutedNodes = nodes.map((node) => {
    // If Dagre failed to position a node, fallback to (0,0)
    const nodeWithPosition = dagreGraph.node(node.id);
    
    return {
      ...node,
      targetPosition: 'top',
      sourcePosition: 'bottom',
      position: {
        x: nodeWithPosition ? nodeWithPosition.x - nodeWidth / 2 : 0,
        y: nodeWithPosition ? nodeWithPosition.y - nodeHeight / 2 : 0,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
};

export default function RLTTree({ data }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    console.log("📊 Raw Data received in RLTTree:", data); // <--- DEBUG LOG

    try {
        if (data && data.nodes && Array.isArray(data.nodes)) {
            console.log(`✅ Found ${data.nodes.length} nodes and ${data.edges?.length || 0} edges.`);
            
            const { nodes: layoutNodes, edges: layoutEdges } = getLayoutedElements(
                data.nodes,
                data.edges
            );
            
            setNodes(layoutNodes);
            setEdges(layoutEdges);
        } else {
            console.warn("⚠️ Data is invalid or empty:", data);
        }
    } catch (error) {
        console.error("🔥 CRASH in Layout Calculation:", error);
    }
  }, [data, setNodes, setEdges]);

  // Fallback UI if no data
  if (!nodes || nodes.length === 0) {
      return (
          <div className="flex items-center justify-center h-full text-gray-400">
              {data ? "Processing Layout..." : "No Data Loaded"}
          </div>
      );
  }

  return (
    <div style={{ height: '100%', width: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.1} // Allow zooming out far
      >
        <Background color="#f1f5f9" gap={20} />
        <Controls />
        <MiniMap 
            nodeColor={n => n.data?.isLeaf ? '#4ade80' : '#60a5fa'} 
            maskColor="rgba(240, 240, 240, 0.6)"
        />
      </ReactFlow>
    </div>
  );
}