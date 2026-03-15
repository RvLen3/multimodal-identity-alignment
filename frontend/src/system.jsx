import React, { useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { SiBilibili, SiTiktok, SiSinaweibo } from 'react-icons/si';
import { Database, GitMerge, Search, Terminal, Activity, Settings } from 'lucide-react';
import { getSystemOverview, getTaskDetail } from './api/client';

const platformIconMap = {
  bilibili: SiBilibili,
  douyin: SiTiktok,
  weibo: SiSinaweibo,
};

function PlatformBadge({ platform, className = '' }) {
  const Icon = platformIconMap[platform] || Activity;
  return <Icon className={className || 'h-3.5 w-3.5 text-stone-500 shrink-0'} />;
}

export default function IdentityAlignmentSystem() {
  const location = useLocation();
  const [stats, setStats] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [selectedTaskId, setSelectedTaskId] = useState(location.state?.taskId || null);
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const overview = await getSystemOverview();
        if (!mounted) return;
        setStats(overview.stats || []);
        setTasks(overview.tasks || []);
        const initialTaskId = location.state?.taskId || overview.selected_task_id || overview.tasks?.[0]?.id || null;
        setSelectedTaskId(initialTaskId);
      } catch (err) {
        if (!mounted) return;
        setErrorMessage(err.message || '加载系统概览失败');
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [location.state?.taskId]);

  useEffect(() => {
    if (!selectedTaskId) {
      setSelectedDetail(null);
      return;
    }
    let mounted = true;
    (async () => {
      try {
        const detail = await getTaskDetail(selectedTaskId);
        if (mounted) {
          setSelectedDetail(detail);
          setErrorMessage('');
        }
      } catch (err) {
        if (mounted) {
          setErrorMessage(err.message || '加载任务详情失败');
        }
      }
    })();
    return () => {
      mounted = false;
    };
  }, [selectedTaskId]);

  const selectedTask = useMemo(
    () => tasks.find((task) => task.id === selectedTaskId) || null,
    [tasks, selectedTaskId]
  );

  return (
    <div className="h-screen w-full bg-stone-100 flex flex-col font-sans text-stone-900 overflow-hidden">
      <header className="h-12 bg-stone-900 text-stone-200 flex items-center justify-between px-4 shrink-0 border-b border-stone-950">
        <div className="flex items-center gap-2 text-stone-50">
          <Database className="w-4 h-4 text-blue-400" />
          <span className="font-bold text-sm tracking-wide">IdentityAlign Engine</span>
          <span className="px-1.5 py-0.5 bg-stone-800 text-stone-400 text-[10px] font-mono rounded-sm border border-stone-700">api-wired</span>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center bg-stone-800 border border-stone-700 rounded-sm px-2 py-1 w-64 focus-within:border-blue-500 transition-colors">
            <Search className="w-3.5 h-3.5 text-stone-400 mr-2" />
            <input
              type="text"
              placeholder="Query ID / Hash..."
              className="bg-transparent border-none outline-none text-xs w-full text-stone-200 placeholder:text-stone-500 font-mono"
            />
          </div>
          <button className="text-stone-400 hover:text-stone-100 transition-colors">
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      <div className="h-10 bg-white border-b border-stone-300 flex items-center px-4 shrink-0 text-xs overflow-x-auto">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-none bg-green-500"></div>
            <span className="font-medium text-stone-500">SYS_STATUS:</span>
            <span className="font-mono font-bold">ONLINE</span>
          </div>
          <div className="w-px h-4 bg-stone-300"></div>
          {stats.map((stat, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <span className="font-medium text-stone-500">{stat.label}:</span>
              <span className="font-mono font-bold">{stat.value}</span>
              <span className="font-mono text-[10px] text-green-600">({stat.trend})</span>
            </div>
          ))}
        </div>
      </div>

      <main className="flex-1 flex overflow-hidden">
        <aside className="w-80 bg-stone-50 border-r border-stone-300 flex flex-col shrink-0">
          <div className="px-3 py-2 bg-stone-200/50 border-b border-stone-300 flex justify-between items-center">
            <h2 className="text-xs font-bold text-stone-700 uppercase tracking-wider">Alignment Queue</h2>
            <span className="text-[10px] font-mono bg-stone-300 px-1.5 py-0.5 rounded-sm text-stone-700">LIVE</span>
          </div>

          <div className="flex-1 overflow-y-auto">
            {tasks.map((task) => {
              const isSelected = selectedTaskId === task.id;
              return (
                <div
                  key={task.id}
                  onClick={() => setSelectedTaskId(task.id)}
                  className={`border-b border-stone-200 p-3 cursor-pointer transition-colors ${
                    isSelected ? 'bg-blue-50 border-l-2 border-l-blue-600' : 'hover:bg-white border-l-2 border-l-transparent'
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-mono text-xs font-semibold text-stone-800">{task.id}</span>
                    <span className="font-mono text-[10px] text-stone-500">{task.timestamp}</span>
                  </div>

                  <div className="flex items-center justify-between mt-1">
                    <div className="flex items-center gap-1.5 flex-1 min-w-0">
                      <PlatformBadge platform={task.targetA.platform} />
                      <span className="text-xs truncate">{task.targetA.name}</span>
                    </div>
                    <GitMerge className="w-3 h-3 text-stone-400 mx-1 shrink-0" />
                    <div className="flex items-center gap-1.5 flex-1 min-w-0 justify-end">
                      <span className="text-xs truncate">{task.targetB.name}</span>
                      <PlatformBadge platform={task.targetB.platform} />
                    </div>
                  </div>

                  <div className="mt-3 flex items-center justify-between">
                    {task.status === 'DONE' && <span className="text-[10px] font-mono px-1 border border-green-300 bg-green-50 text-green-700">DONE</span>}
                    {task.status === 'SYNC' && <span className="text-[10px] font-mono px-1 border border-blue-300 bg-blue-50 text-blue-700">SYNCING</span>}
                    {task.status === 'FAIL' && <span className="text-[10px] font-mono px-1 border border-red-300 bg-red-50 text-red-700">REJECTED</span>}
                    {task.score !== null ? (
                      <span className="font-mono text-xs font-bold text-stone-700">Conf: {task.score.toFixed(3)}</span>
                    ) : (
                      <span className="font-mono text-xs text-stone-400">Computing...</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </aside>

        <div className="flex-1 flex flex-col bg-white overflow-hidden">
          {loading ? (
            <div className="flex-1 flex items-center justify-center text-stone-500 font-mono text-sm">LOADING_OVERVIEW...</div>
          ) : errorMessage ? (
            <div className="flex-1 flex items-center justify-center text-red-600 font-mono text-sm">API_ERROR: {errorMessage}</div>
          ) : selectedTask && selectedDetail ? (
            <>
              <div className="px-6 py-4 border-b border-stone-300 bg-stone-50 flex items-center justify-between shrink-0">
                <div>
                  <div className="flex items-center gap-3">
                    <h1 className="text-lg font-bold text-stone-900">Alignment Record</h1>
                    <span className="font-mono text-sm bg-stone-200 px-2 py-0.5 border border-stone-300 rounded-sm">{selectedDetail.id}</span>
                  </div>
                  <p className="text-xs text-stone-500 mt-1 font-mono">Job Execution: {selectedTask.timestamp} | Node: API-Placeholder</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-stone-500 uppercase tracking-wider mb-1">Global Confidence</p>
                  <p className="font-mono text-2xl font-bold text-green-600">{selectedDetail.overallScore.toFixed(4)}</p>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
                <div className="border border-stone-300 rounded-sm overflow-hidden bg-white grid grid-cols-2 divide-x divide-stone-300">
                  <div className="p-5">
                    <div className="mb-3 text-xs text-stone-500 uppercase">Entity A</div>
                    <p className="font-mono text-sm">@{selectedDetail.profileA.username}</p>
                    <p className="text-xs text-stone-500 mt-1">{selectedDetail.profileA.platform}</p>
                    <p className="text-xs text-stone-600 mt-3">{selectedDetail.profileA.bio}</p>
                  </div>
                  <div className="p-5 bg-stone-50/50">
                    <div className="mb-3 text-xs text-stone-500 uppercase">Entity B</div>
                    <p className="font-mono text-sm">@{selectedDetail.profileB.username}</p>
                    <p className="text-xs text-stone-500 mt-1">{selectedDetail.profileB.platform}</p>
                    <p className="text-xs text-stone-600 mt-3">{selectedDetail.profileB.bio}</p>
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-bold text-stone-800 mb-3 flex items-center gap-2">
                    <Database className="w-4 h-4" />
                    Feature Alignment Matrix
                  </h3>
                  <div className="border border-stone-300 rounded-sm overflow-hidden">
                    <table className="w-full text-left text-sm border-collapse">
                      <thead className="bg-stone-100 text-xs uppercase text-stone-500">
                        <tr>
                          <th className="p-3 border-b border-stone-300 font-semibold">Modality</th>
                          <th className="p-3 border-b border-stone-300 font-semibold">Details</th>
                          <th className="p-3 border-b border-stone-300 font-semibold">Weight</th>
                          <th className="p-3 border-b border-stone-300 font-semibold text-right">Score</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-stone-200">
                        {selectedDetail.modalities.map((mod, idx) => (
                          <tr key={idx} className="hover:bg-stone-50/50">
                            <td className="p-3 font-mono text-xs text-stone-800">{mod.name}</td>
                            <td className="p-3 text-xs text-stone-600">{mod.desc}</td>
                            <td className="p-3 font-mono text-xs text-stone-500">W_{mod.weight}</td>
                            <td className="p-3 font-mono text-sm font-bold text-right text-stone-900">{mod.score.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="border border-stone-300 rounded-sm overflow-hidden bg-stone-900 text-stone-300">
                  <div className="bg-stone-800 px-3 py-1.5 border-b border-stone-700 flex items-center gap-2">
                    <Terminal className="w-3.5 h-3.5" />
                    <span className="text-[10px] font-mono uppercase">System Decision Output</span>
                  </div>
                  <div className="p-4 font-mono text-xs leading-relaxed">
                    {selectedDetail.decision_lines.map((line, idx) => (
                      <p key={idx}>{line}</p>
                    ))}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-stone-400 flex-col gap-3">
              <Activity className="w-8 h-8 opacity-20" />
              <p className="font-mono text-sm">SELECT_TASK_TO_VIEW_DETAILS</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
