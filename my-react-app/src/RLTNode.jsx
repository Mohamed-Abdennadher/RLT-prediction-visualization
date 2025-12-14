/* eslint-disable react/prop-types */
import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
// 1. Remove ResponsiveContainer from imports
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

const RLTNode = ({ data }) => {
  const isLeaf = data.isLeaf;

  return (
    // Fixed width card
    <div className="rlt-node-card" style={{ width: '280px', background: 'white', borderRadius: '8px', border: '1px solid #ccc', overflow: 'hidden' }}>
      
      <Handle type="target" position={Position.Top} style={{ background: '#555' }} />
      
      {/* HEADER */}
      <div style={{ padding: '10px', background: isLeaf ? '#f0fdf4' : '#eff6ff', borderBottom: '1px solid #eee' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
             <span style={{ fontSize: '10px', fontWeight: 'bold', textTransform: 'uppercase', color: '#9ca3af' }}>
               {isLeaf ? 'Leaf Node' : 'Split Node'}
             </span>
             <span style={{ fontSize: '10px', background: '#e5e7eb', padding: '2px 6px', borderRadius: '10px' }}>
               N={data.samples}
             </span>
        </div>
        <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#1f2937' }}>
           {isLeaf ? (
             <span>🎯 {data.label}</span>
           ) : (
             <span>
               <span style={{ color: '#2563eb' }}>{data.splitFeature}</span>
               <span style={{ margin: '0 4px', color: '#6b7280' }}>≤</span>
               <span>{data.label.split(': ')[1]}</span>
             </span>
           )}
        </div>
      </div>

      {/* BODY */}
      <div style={{ padding: '10px' }}>
        
        {/* PILOT CHART - FIXED DIMENSIONS */}
        {data.pilotData && data.pilotData.length > 0 && (
          <div style={{ marginTop: '5px' }}>
            <p style={{ fontSize: '10px', fontWeight: 'bold', color: '#9ca3af', textTransform: 'uppercase', marginBottom: '5px' }}>
              Pilot Importance
            </p>
            
            {/* 
               FIX: Removed ResponsiveContainer. 
               We use explicit width={250} and height={110} 
            */}
            <BarChart 
                width={250} 
                height={110} 
                data={data.pilotData} 
                layout="vertical" 
                margin={{ left: 0, right: 0, top: 0, bottom: 0 }}
            >
              <XAxis type="number" hide />
              <YAxis 
                dataKey="name" 
                type="category" 
                width={70} 
                tick={{fontSize: 10}} 
                interval={0}
              />
              <Tooltip 
                cursor={{fill: 'transparent'}}
                contentStyle={{ fontSize: '11px' }}
              />
              <Bar dataKey="value" barSize={12} radius={[0, 4, 4, 0]}>
                {data.pilotData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill="#6366f1" />
                ))}
              </Bar>
            </BarChart>
          </div>
        )}

        {/* MUTED VARS */}
        {data.mutedVars && data.mutedVars.length > 0 && (
          <div style={{ marginTop: '10px' }}>
             <p style={{ fontSize: '10px', fontWeight: 'bold', color: '#f87171', textTransform: 'uppercase' }}>
               🚫 Muted
             </p>
             <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
               {data.mutedVars.slice(0, 3).map((v, i) => (
                 <span key={i} style={{ fontSize: '10px', background: '#fef2f2', color: '#dc2626', padding: '2px 4px', borderRadius: '4px', border: '1px solid #fee2e2' }}>
                   {v}
                 </span>
               ))}
               {data.mutedVars.length > 3 && (
                 <span style={{ fontSize: '10px', color: '#9ca3af' }}>+{data.mutedVars.length - 3}</span>
               )}
             </div>
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} style={{ background: '#555' }} />
    </div>
  );
};

export default memo(RLTNode);