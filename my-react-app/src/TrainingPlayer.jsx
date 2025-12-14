/* eslint-disable react/prop-types */
import React, { useState, useEffect, useCallback } from 'react';
import ReactFlow, { 
  useNodesState, 
  useEdgesState, 
  Controls, 
  Background, 
  ReactFlowProvider,
  useReactFlow // <--- NEED THIS FOR SMOOTH CAMERA
} from 'reactflow';
import dagre from 'dagre';
import 'reactflow/dist/style.css'; 
import RLTNode from './RLTNode'; 

const nodeTypes = { rltNode: RLTNode };

// --- 1. CSS FOR SMOOTH ANIMATION ---
// We inject this style into the component to force React Flow nodes to glide
const transitionStyles = `
  /* Animate the movement  nodes */
  .react-flow__node {
    transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
  }
  
  /* Animate the appearing of nodes */
  @keyframes popIn {
    0% { transform: scale(0); opacity: 0; }
    80% { transform: scale(1.1); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
  }

  .rlt-node-card {
    animation: popIn 0.4s ease-out forwards;
    transition: all 0.3s ease; /* For border color changes */
  }
`;

const styles = {
  container: {
    position: 'absolute',
    top: '20px',
    left: '85%',
    transform: 'translateX(-50%)',
    zIndex: 50,
    backgroundColor: 'rgba(255, 255, 255, 0.85)',
    backdropFilter: 'blur(12px)',
    padding: '10px',
    borderRadius: '20px',
    boxShadow: '0 20px 40px -5px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(0,0,0,0.05)',
    width: '350px',
    height:'150px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  },
  headerRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: '4px',
  },
  title: {
    fontSize: '16px',
    fontWeight: '800',
    background: 'linear-gradient(to right, #4f46e5, #9333ea)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    margin: 0,
    letterSpacing: '-0.02em',
  },
  stepCounter: {
    fontSize: '12px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    color: '#6b7280',
    background: '#f3f4f6',
    padding: '4px 10px',
    borderRadius: '20px',
    border: '1px solid #e5e7eb',
  },
  controlsRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  },
  playButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '44px',
    height: '44px',
    borderRadius: '50%',
    border: 'none',
    cursor: 'pointer',
    color: 'white',
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    boxShadow: '0 4px 12px rgba(79, 70, 229, 0.3)',
  },
  playButtonActive: {
     transform: 'scale(0.95)',
     boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.1)',
  },
  slider: {
    flex: 1,
    cursor: 'pointer',
    accentColor: '#4f46e5',
    height: '6px',
    borderRadius: '10px',
  },
  infoBox: {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '16px',
    boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.02)',
    border: '1px solid #f3f4f6',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    minHeight: '80px',
    justifyContent: 'center'
  },
  badge: {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '4px 10px',
    borderRadius: '6px',
    fontSize: '11px',
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'white',
    width: 'fit-content',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  }
};

const getEventColor = (type) => {
    switch(type) {
        case 'NODE_INIT': return '#6b7280'; 
        case 'PILOT_RUN': return '#8b5cf6'; 
        case 'MUTE_VARS': return '#ef4444'; 
        case 'SPLIT_DECISION': return '#3b82f6'; 
        case 'MAKE_LEAF': return '#10b981'; 
        default: return '#6b7280';
    }
};

const getLayout = (nodes, edges) => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: 'TB' });

  // Increased spacing for smoother look
  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: 320, height: 350 });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  return nodes.map((node) => {
    const pos = dagreGraph.node(node.id);
    return {
      ...node,
      position: { 
          x: pos ? pos.x - 160 : 0, 
          y: pos ? pos.y - 175 : 0 
      }
    };
  });
};

function TrainingPlayerInner({ history }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  // --- SMOOTH CAMERA HOOK ---
  const { fitView } = useReactFlow();

  const renderStep = useCallback((stepIndex) => {
    if (!history || history.length === 0) return;

    let tempNodes = [];
    let tempEdges = [];

    for (let i = 0; i <= stepIndex; i++) {
        const event = history[i];
        
        if (event.type === 'NODE_INIT') {
            tempNodes.push({
                id: event.nodeId,
                type: 'rltNode',
                data: { 
                    label: 'Analyzing...', 
                    samples: event.samples,
                    isLeaf: false,
                    pilotData: [],
                    mutedVars: []
                },
                position: { x: 0, y: 0 } // Layout handles this
            });

            if (event.parentId) {
                tempEdges.push({
                    id: `e${event.parentId}-${event.nodeId}`,
                    source: event.parentId,
                    target: event.nodeId,
                    type: 'smoothstep',
                    animated: true
                });
            }
        }
        else if (event.type === 'PILOT_RUN') {
            const nodeIndex = tempNodes.findIndex(n => n.id === event.nodeId);
            if (nodeIndex !== -1) {
                tempNodes[nodeIndex].data.pilotData = event.pilotData;
                tempNodes[nodeIndex].style = { 
                    border: '2px solid #8b5cf6', 
                    boxShadow: '0 0 20px rgba(139, 92, 246, 0.3)',
                    transition : 0.3
                };
            }
        }
        else if (event.type === 'MUTE_VARS') {
            const nodeIndex = tempNodes.findIndex(n => n.id === event.nodeId);
            if (nodeIndex !== -1) {
                tempNodes[nodeIndex].data.mutedVars = event.mutedVars;
            }
        }
        else if (event.type === 'SPLIT_DECISION') {
            const nodeIndex = tempNodes.findIndex(n => n.id === event.nodeId);
            if (nodeIndex !== -1) {
                tempNodes[nodeIndex].data.label = event.label;
                tempNodes[nodeIndex].data.splitFeature = event.splitFeature;
                tempNodes[nodeIndex].style = undefined; 
            }
        }
        else if (event.type === 'MAKE_LEAF') {
            const nodeIndex = tempNodes.findIndex(n => n.id === event.nodeId);
            if (nodeIndex !== -1) {
                tempNodes[nodeIndex].data.label = event.label;
                tempNodes[nodeIndex].data.isLeaf = true;
                tempNodes[nodeIndex].style = undefined;
            }
        }
    }

    const layouted = getLayout(tempNodes, tempEdges);
    setNodes(layouted);
    setEdges(tempEdges);
    
    // --- SMOOTH CAMERA FOLLOW ---
    // We delay slightly to let the nodes calculate position, then smooth pan
    window.requestAnimationFrame(() => {
        fitView({ 
            duration: 800,  // 800ms smooth slide
            padding: 0.2, 
            minZoom: 0.1, 
            maxZoom: 1.5 
        });
    });

  }, [history, setNodes, setEdges, fitView]);

  useEffect(() => {
    let interval;
    if (isPlaying && currentStep < history.length - 1) {
      interval = setInterval(() => {
        setCurrentStep(prev => prev + 1);
      }, 600); // SLOWED DOWN to 1.2s to let animation breathe
    } else if (currentStep >= history.length - 1) {
        setIsPlaying(false);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentStep, history.length]);

  useEffect(() => {
    if(history && history.length > 0) {
        renderStep(currentStep);
    }
  }, [currentStep, renderStep, history]);

  const currentEvent = history[currentStep] || {};

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {/* Inject CSS Styles */}
      <style>{transitionStyles}</style>

      <div style={styles.container}>
        <div style={styles.headerRow}>
            <h3 style={styles.title}>Training Process</h3>
            <span style={styles.stepCounter}>
                {currentStep + 1} / {history.length}
            </span>
        </div>

        <div style={styles.infoBox}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span style={{ ...styles.badge, backgroundColor: getEventColor(currentEvent.type) }}>
                    {currentEvent.type?.replace('_', ' ')}
                </span>
                <span style={{ fontSize: '14px', fontWeight: '600', color: '#111827' }}>
                    Node {currentEvent.nodeId}
                </span>
            </div>
            
            <div style={{ color: '#4b5563', fontSize: '13px', lineHeight: '1.5' }}>
                {currentEvent.type === 'NODE_INIT' && `Initialized with ${currentEvent.samples} samples. Checking purity...`}
                {currentEvent.type === 'PILOT_RUN' && "⚡ Running Embedded Random Forest to calculate variable importance scores..."}
                {currentEvent.type === 'MUTE_VARS' && `🚫 Muting ${currentEvent.mutedVars?.length} noisy variables based on pilot scores.`}
                {currentEvent.type === 'SPLIT_DECISION' && <span>Best split found: <b>{currentEvent.splitFeature}</b> ≤ {currentEvent.label.split(': ')[1]}</span>}
                {currentEvent.type === 'MAKE_LEAF' && `🛑 Stopping criteria met. Final Prediction: ${currentEvent.label.split(': ')[1]}`}
            </div>
        </div>
        
        <div style={styles.controlsRow}>
            <button 
                onClick={() => setIsPlaying(!isPlaying)}
                style={{
                    ...styles.playButton,
                    backgroundColor: isPlaying ? '#ef4444' : '#4f46e5'
                }}
            >
                {isPlaying ? (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/></svg>
                ) : (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                )}
            </button>

            <input 
                type="range" 
                min="0" 
                max={history.length - 1} 
                value={currentStep} 
                onChange={(e) => {
                    setIsPlaying(false);
                    setCurrentStep(parseInt(e.target.value));
                }}
                style={styles.slider}
            />
        </div>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        // Removed static fitView prop, we handle it manually
        minZoom={0.1}
      >
        <Background color="#f8fafc" gap={24} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

// Wrapper needed for useReactFlow hook
export default function TrainingPlayer(props) {
    return (
        <ReactFlowProvider>
            <TrainingPlayerInner {...props} />
        </ReactFlowProvider>
    )
}