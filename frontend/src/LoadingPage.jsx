import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { SiBilibili, SiTiktok, SiSinaweibo } from 'react-icons/si';
import {
  ArrowRight,
  Check,
  Cpu,
  Fingerprint,
  LayoutDashboard,
  Radar,
  Crosshair,
  ShieldCheck,
  Network,
  FileText,
  Image as ImageIcon,
  Activity,
  BarChart3,
  Sparkles,
} from 'lucide-react';
import { getUserSuggestions, searchIdentity, verifyIdentity } from './api/client';

const platformIconMap = {
  bili: SiBilibili,
  bilibili: SiBilibili,
  douyin: SiTiktok,
  weibo: SiSinaweibo,
};

const platformOptions = [
  { id: 'bili', label: 'Bilibili' },
  { id: 'douyin', label: 'Douyin' },
  { id: 'weibo', label: 'Weibo' },
];

const features = [
  {
    icon: ImageIcon,
    title: '视觉信号建模',
    desc: '将头像、封面、背景图映射到统一向量空间，提升跨平台识别稳定性。',
  },
  {
    icon: FileText,
    title: '语义与文体分析',
    desc: '对简介、标题、内容文本进行统一语义表示，捕捉长期表达习惯。',
  },
  {
    icon: Activity,
    title: '行为时序特征',
    desc: '整合活跃周期与互动节奏，增强复杂场景下的身份判断可信度。',
  },
  {
    icon: Network,
    title: '关系图谱推断',
    desc: '结合关系网络结构相似度，形成更可解释的多证据决策结果。',
  },
];

function buildProfileUrl(platform, uid) {
  const p = (platform || '').toLowerCase();
  const id = encodeURIComponent(uid || '');
  if (p === 'bili' || p === 'bilibili') return `https://space.bilibili.com/${id}`;
  if (p === 'douyin') return `https://www.douyin.com/user/${id}`;
  if (p === 'weibo') return `https://weibo.com/u/${id}`;
  return '';
}

export default function LandingPage() {
  const navigate = useNavigate();
  const [demoMode, setDemoMode] = useState('search');
  const [demoState, setDemoState] = useState('idle');

  const [selectedPlatforms, setSelectedPlatforms] = useState(['bili', 'douyin']);
  const [searchSourcePlatform, setSearchSourcePlatform] = useState('douyin');
  const [searchUserId, setSearchUserId] = useState('AlexChen_Douyin');
  const [searchSuggestions, setSearchSuggestions] = useState([]);
  const [showSearchSuggestions, setShowSearchSuggestions] = useState(false);

  const [verifySourcePlatform, setVerifySourcePlatform] = useState('douyin');
  const [verifySourceUserId, setVerifySourceUserId] = useState('AlexChen_Douyin');
  const [verifySourceSuggestions, setVerifySourceSuggestions] = useState([]);
  const [showVerifySourceSuggestions, setShowVerifySourceSuggestions] = useState(false);

  const [verifyTargetPlatform, setVerifyTargetPlatform] = useState('bili');
  const [verifyTargetUserId, setVerifyTargetUserId] = useState('AlexChen_Bili');
  const [verifyTargetSuggestions, setVerifyTargetSuggestions] = useState([]);
  const [showVerifyTargetSuggestions, setShowVerifyTargetSuggestions] = useState(false);

  const [searchResult, setSearchResult] = useState(null);
  const [verifyResult, setVerifyResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [latestTaskId, setLatestTaskId] = useState(null);

  const searchSource = {
    platform: searchSourcePlatform,
    account: searchUserId.trim(),
  };

  const resetDemo = () => {
    setDemoState('idle');
    setSearchResult(null);
    setVerifyResult(null);
    setErrorMessage('');
    setShowSearchSuggestions(false);
    setShowVerifySourceSuggestions(false);
    setShowVerifyTargetSuggestions(false);
  };

  const fetchSuggestions = async (platform, q, setList, setVisible) => {
    if (!q) {
      setList([]);
      setVisible(false);
      return;
    }
    try {
      const res = await getUserSuggestions(platform, q, 8);
      setList(res.items || []);
      setVisible(true);
    } catch {
      setList([]);
    }
  };

  useEffect(() => {
    if (demoMode !== 'search') {
      setShowSearchSuggestions(false);
      return;
    }

    const q = searchUserId.trim();
    const timer = setTimeout(() => {
      fetchSuggestions(searchSourcePlatform, q, setSearchSuggestions, setShowSearchSuggestions);
    }, 160);

    return () => clearTimeout(timer);
  }, [demoMode, searchSourcePlatform, searchUserId]);

  useEffect(() => {
    if (demoMode !== 'verify') {
      setShowVerifySourceSuggestions(false);
      return;
    }

    const q = verifySourceUserId.trim();
    const timer = setTimeout(() => {
      fetchSuggestions(verifySourcePlatform, q, setVerifySourceSuggestions, setShowVerifySourceSuggestions);
    }, 160);

    return () => clearTimeout(timer);
  }, [demoMode, verifySourcePlatform, verifySourceUserId]);

  useEffect(() => {
    if (demoMode !== 'verify') {
      setShowVerifyTargetSuggestions(false);
      return;
    }

    const q = verifyTargetUserId.trim();
    const timer = setTimeout(() => {
      fetchSuggestions(verifyTargetPlatform, q, setVerifyTargetSuggestions, setShowVerifyTargetSuggestions);
    }, 160);

    return () => clearTimeout(timer);
  }, [demoMode, verifyTargetPlatform, verifyTargetUserId]);

  const togglePlatform = (platform) => {
    if (selectedPlatforms.includes(platform)) {
      if (selectedPlatforms.length > 1) {
        setSelectedPlatforms(selectedPlatforms.filter((p) => p !== platform));
      }
      return;
    }
    setSelectedPlatforms([...selectedPlatforms, platform]);
  };

  const handleDemoRun = async () => {
    setErrorMessage('');
    setDemoState('computing');
    try {
      if (demoMode === 'search') {
        if (!searchSource.account) {
          setDemoState('idle');
          setErrorMessage('请输入有效用户 ID');
          return;
        }
        const res = await searchIdentity({
          source: searchSource,
          target_platforms: selectedPlatforms,
          top_k: 5,
        });
        setSearchResult(res);
        setLatestTaskId(res.task_id);
        setShowSearchSuggestions(false);
      } else {
        const source = {
          platform: verifySourcePlatform,
          account: verifySourceUserId.trim(),
        };
        const target = {
          platform: verifyTargetPlatform,
          account: verifyTargetUserId.trim(),
        };

        if (!source.account || !target.account) {
          setDemoState('idle');
          setErrorMessage('请输入待核验双方的用户 ID');
          return;
        }

        const res = await verifyIdentity({ source, target });
        setVerifyResult(res);
        setLatestTaskId(res.task_id);
        setShowVerifySourceSuggestions(false);
        setShowVerifyTargetSuggestions(false);
      }
      setDemoState('done');
    } catch (error) {
      setDemoState('idle');
      setErrorMessage(error.message || '接口调用失败');
    }
  };

  const navigateToSystem = () => {
    navigate('/system', { state: { taskId: latestTaskId } });
  };

  return (
    <div className="min-h-screen bg-[#f7f8fb] text-slate-900 [font-family:'Sora','Noto_Sans_SC',sans-serif]">
      <header className="sticky top-0 z-40 border-b border-slate-200/80 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-900 text-white shadow-sm">
              <Fingerprint className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold tracking-wide text-slate-900">IdentityAlign Engine</p>
              <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Multimodal Identity Matching</p>
            </div>
          </div>
          <button
            onClick={navigateToSystem}
            className="inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-800 transition hover:bg-slate-50"
          >
            <LayoutDashboard className="h-4 w-4" />
            进入管理控制台
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-14 md:py-16">
        <section className="grid items-start gap-12 lg:grid-cols-[1.15fr_0.85fr]">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-medium text-blue-700">
              <Sparkles className="h-3.5 w-3.5" />
              统一身份对齐引擎
            </div>

            <h1 className="mt-6 text-4xl font-semibold leading-tight text-slate-950 md:text-5xl lg:text-6xl">
              更可靠地连接
              <br />
              跨平台数字身份
            </h1>

            <p className="mt-6 max-w-2xl text-base leading-relaxed text-slate-600 md:text-lg">
              面向风控、治理与研究场景，IdentityAlign 通过视觉、语义、行为、图谱四维融合，
              提供可解释的身份检索与核验能力。当前页面已对接后端 API，可直接联调演示。
            </p>

            <div className="mt-8 grid gap-4 sm:grid-cols-3">
              <div className="rounded-2xl border border-slate-200 bg-white p-4">
                <p className="text-xs uppercase tracking-wider text-slate-500">Node Scale</p>
                <p className="mt-1 text-2xl font-semibold text-slate-900">10B+</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-4">
                <p className="text-xs uppercase tracking-wider text-slate-500">API Status</p>
                <p className="mt-1 text-2xl font-semibold text-slate-900">Online</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-4">
                <p className="text-xs uppercase tracking-wider text-slate-500">Response</p>
                <p className="mt-1 text-2xl font-semibold text-slate-900">&lt;2s</p>
              </div>
            </div>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <button
                onClick={navigateToSystem}
                className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-6 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
              >
                进入系统页
                <ArrowRight className="h-4 w-4" />
              </button>
              <span className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-xs font-medium text-emerald-700">
                <ShieldCheck className="h-4 w-4" />
                API Contract Ready
              </span>
            </div>
          </div>

          <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-5 grid grid-cols-2 gap-2 rounded-xl bg-slate-100 p-1">
              <button
                onClick={() => {
                  setDemoMode('search');
                  resetDemo();
                }}
                className={`rounded-lg px-3 py-2 text-sm font-medium transition ${demoMode === 'search' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
              >
                <span className="inline-flex items-center gap-2">
                  <Radar className="h-4 w-4" /> 1 查 N
                </span>
              </button>
              <button
                onClick={() => {
                  setDemoMode('verify');
                  resetDemo();
                }}
                className={`rounded-lg px-3 py-2 text-sm font-medium transition ${demoMode === 'verify' ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
              >
                <span className="inline-flex items-center gap-2">
                  <Crosshair className="h-4 w-4" /> 1 对 1
                </span>
              </button>
            </div>

            {demoMode === 'search' && (
              <div className="space-y-4">
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="mb-1 text-[11px] uppercase tracking-wider text-slate-500">Source Platform</p>
                  <div className="flex flex-wrap gap-2">
                    {platformOptions.map(({ id: platform, label }) => {
                      const Icon = platformIconMap[platform];
                      const selected = searchSourcePlatform === platform;
                      return (
                        <button
                          key={platform}
                          onClick={() => demoState === 'idle' && setSearchSourcePlatform(platform)}
                          className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                            selected
                              ? 'border-blue-200 bg-blue-50 text-blue-700'
                              : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                          }`}
                        >
                          <Icon className="h-3.5 w-3.5" />
                          {label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="mb-1 text-[11px] uppercase tracking-wider text-slate-500">Source User ID</p>
                  <div className="relative">
                    <input
                      type="text"
                      value={searchUserId}
                      onChange={(e) => setSearchUserId(e.target.value)}
                      onFocus={() => setShowSearchSuggestions(true)}
                      onBlur={() => setTimeout(() => setShowSearchSuggestions(false), 120)}
                      disabled={demoState === 'computing'}
                      placeholder="请输入来源平台用户 ID"
                      className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-blue-300 focus:ring-2 focus:ring-blue-100"
                    />
                    {showSearchSuggestions && searchSuggestions.length > 0 && (
                      <div className="absolute z-20 mt-1 max-h-44 w-full overflow-y-auto rounded-lg border border-slate-200 bg-white shadow-lg">
                        {searchSuggestions.map((uid) => (
                          <button
                            key={uid}
                            type="button"
                            onMouseDown={(e) => {
                              e.preventDefault();
                              setSearchUserId(uid);
                              setShowSearchSuggestions(false);
                            }}
                            className="block w-full border-b border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 last:border-b-0"
                          >
                            {uid}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div>
                  <p className="mb-2 text-[11px] uppercase tracking-wider text-slate-500">Target Platforms</p>
                  <div className="flex flex-wrap gap-2">
                    {platformOptions.map(({ id: platform, label }) => {
                      const Icon = platformIconMap[platform];
                      const selected = selectedPlatforms.includes(platform);
                      return (
                        <button
                          key={platform}
                          onClick={() => demoState === 'idle' && togglePlatform(platform)}
                          className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                            selected
                              ? 'border-blue-200 bg-blue-50 text-blue-700'
                              : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                          }`}
                        >
                          <Icon className="h-3.5 w-3.5" />
                          {label}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {demoMode === 'verify' && (
              <div className="space-y-4">
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="mb-1 text-[11px] uppercase tracking-wider text-slate-500">Source Platform</p>
                  <div className="flex flex-wrap gap-2">
                    {platformOptions.map(({ id: platform, label }) => {
                      const Icon = platformIconMap[platform];
                      const selected = verifySourcePlatform === platform;
                      return (
                        <button
                          key={`verify-source-${platform}`}
                          onClick={() => demoState === 'idle' && setVerifySourcePlatform(platform)}
                          className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                            selected
                              ? 'border-indigo-200 bg-indigo-50 text-indigo-700'
                              : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                          }`}
                        >
                          <Icon className="h-3.5 w-3.5" />
                          {label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="mb-1 text-[11px] uppercase tracking-wider text-slate-500">Source User ID</p>
                  <div className="relative">
                    <input
                      type="text"
                      value={verifySourceUserId}
                      onChange={(e) => setVerifySourceUserId(e.target.value)}
                      onFocus={() => setShowVerifySourceSuggestions(true)}
                      onBlur={() => setTimeout(() => setShowVerifySourceSuggestions(false), 120)}
                      disabled={demoState === 'computing'}
                      placeholder="请输入来源用户 ID"
                      className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-indigo-300 focus:ring-2 focus:ring-indigo-100"
                    />
                    {showVerifySourceSuggestions && verifySourceSuggestions.length > 0 && (
                      <div className="absolute z-20 mt-1 max-h-44 w-full overflow-y-auto rounded-lg border border-slate-200 bg-white shadow-lg">
                        {verifySourceSuggestions.map((uid) => (
                          <button
                            key={`verify-source-${uid}`}
                            type="button"
                            onMouseDown={(e) => {
                              e.preventDefault();
                              setVerifySourceUserId(uid);
                              setShowVerifySourceSuggestions(false);
                            }}
                            className="block w-full border-b border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 last:border-b-0"
                          >
                            {uid}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="mb-1 text-[11px] uppercase tracking-wider text-slate-500">Target Platform</p>
                  <div className="flex flex-wrap gap-2">
                    {platformOptions.map(({ id: platform, label }) => {
                      const Icon = platformIconMap[platform];
                      const selected = verifyTargetPlatform === platform;
                      return (
                        <button
                          key={`verify-target-${platform}`}
                          onClick={() => demoState === 'idle' && setVerifyTargetPlatform(platform)}
                          className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                            selected
                              ? 'border-indigo-200 bg-indigo-50 text-indigo-700'
                              : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                          }`}
                        >
                          <Icon className="h-3.5 w-3.5" />
                          {label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="mb-1 text-[11px] uppercase tracking-wider text-slate-500">Target User ID</p>
                  <div className="relative">
                    <input
                      type="text"
                      value={verifyTargetUserId}
                      onChange={(e) => setVerifyTargetUserId(e.target.value)}
                      onFocus={() => setShowVerifyTargetSuggestions(true)}
                      onBlur={() => setTimeout(() => setShowVerifyTargetSuggestions(false), 120)}
                      disabled={demoState === 'computing'}
                      placeholder="请输入目标用户 ID"
                      className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-indigo-300 focus:ring-2 focus:ring-indigo-100"
                    />
                    {showVerifyTargetSuggestions && verifyTargetSuggestions.length > 0 && (
                      <div className="absolute z-20 mt-1 max-h-44 w-full overflow-y-auto rounded-lg border border-slate-200 bg-white shadow-lg">
                        {verifyTargetSuggestions.map((uid) => (
                          <button
                            key={`verify-target-${uid}`}
                            type="button"
                            onMouseDown={(e) => {
                              e.preventDefault();
                              setVerifyTargetUserId(uid);
                              setShowVerifyTargetSuggestions(false);
                            }}
                            className="block w-full border-b border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 last:border-b-0"
                          >
                            {uid}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            <div className="mt-6">
              {demoState === 'idle' && (
                <button
                  onClick={handleDemoRun}
                  className="w-full rounded-xl bg-slate-900 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
                >
                  {demoMode === 'search' ? '启动跨平台检索' : '执行身份核验'}
                </button>
              )}

              {demoState === 'computing' && (
                <div className="rounded-xl border border-blue-200 bg-blue-50 p-4 text-xs font-mono text-blue-700">
                  PROCESSING_API_REQUEST...
                </div>
              )}

              {demoState === 'done' && demoMode === 'search' && searchResult && (
                <div className="space-y-2">
                  <p className="text-xs font-mono text-emerald-700">FOUND {searchResult.found_count} MATCHES</p>
                  {searchResult.candidates.map((item, idx) => {
                    const Icon = platformIconMap[item.platform] || Radar;
                    const profileUrl = buildProfileUrl(item.platform, item.account);
                    return (
                      <div key={`${item.platform}-${idx}`} className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
                        <Icon className="h-5 w-5 text-slate-700" />
                        <div className="flex-1">
                          <p className="text-sm font-semibold text-slate-900">@{item.account}</p>
                          <p className="text-xs text-slate-500">score: {(item.score * 100).toFixed(1)}%</p>
                        </div>
                        {profileUrl ? (
                          <a
                            href={profileUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100"
                          >
                            查看主页
                          </a>
                        ) : null}
                        <BarChart3 className="h-4 w-4 text-slate-400" />
                      </div>
                    );
                  })}
                  <button
                    onClick={resetDemo}
                    className="mt-2 w-full rounded-xl border border-slate-200 bg-white py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
                  >
                    继续检索
                  </button>
                </div>
              )}

              {demoState === 'done' && demoMode === 'verify' && verifyResult && (
                <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-center">
                  <div className="mb-2 inline-flex items-center gap-2 text-emerald-700">
                    <Check className="h-5 w-5" />
                    <span className="font-semibold">{verifyResult.is_match ? '验证成功：同一实体' : '验证失败：非同一实体'}</span>
                  </div>
                  <p className="text-sm text-emerald-700">综合置信度 {(verifyResult.confidence * 100).toFixed(1)}%</p>
                  <button onClick={navigateToSystem} className="mt-3 text-xs text-blue-700 underline underline-offset-2">
                    在控制台查看详情
                  </button>
                </div>
              )}

              {errorMessage && (
                <div className="mt-3 rounded-xl border border-red-200 bg-red-50 p-3 text-xs font-mono text-red-700">
                  API_ERROR: {errorMessage}
                </div>
              )}
            </div>
          </section>
        </section>

        <section className="mt-16 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((item) => (
            <article key={item.title} className="rounded-2xl border border-slate-200 bg-white p-5">
              <div className="mb-3 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-slate-700">
                <item.icon className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-semibold text-slate-900">{item.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">{item.desc}</p>
            </article>
          ))}
        </section>

        <section className="mt-12 rounded-3xl border border-slate-200 bg-white p-6 md:p-8">
          <div className="grid gap-6 md:grid-cols-[1fr_auto] md:items-center">
            <div>
              <h2 className="text-2xl font-semibold text-slate-900">准备进入系统查看任务流？</h2>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                首页演示已与后端接口对齐。你可以直接进入系统页查看任务队列、详情矩阵与判定输出。
              </p>
              <div className="mt-4 flex flex-wrap gap-2 text-xs">
                <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-slate-600">
                  <Cpu className="h-3.5 w-3.5" /> API Wired
                </span>
                <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-slate-600">
                  <ShieldCheck className="h-3.5 w-3.5" /> Explainable
                </span>
              </div>
            </div>
            <button
              onClick={navigateToSystem}
              className="inline-flex items-center justify-center gap-2 rounded-full bg-slate-900 px-6 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
            >
              进入系统页
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}
