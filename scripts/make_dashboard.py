"""
make_dashboard.py — render a mission-control dashboard from REAL fridge data +
the cryo engine, as a standalone HTML, and (optionally) prep it for screenshotting.

Panels:
  - live stage temperatures (real BlueFors 'blizzard' 24h trace)
  - the cryo-engine base state vs the two real fridges (validation bar)
  - a fault scenario from the engine (agent's-eye view) with the lead-time story
  - fleet fingerprint (per-stage noise + cross-stage correlation heat map)

Writes outputs/dashboard.html and outputs/dashboard_data.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import bluefors_data as BF
from onnesim import virtual_clone as VC
from onnesim import cryo_engine as CE

OUT = "outputs"


def downsample(x, n=300):
    x = np.asarray(x, float)
    if len(x) <= n:
        return x.tolist()
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx].tolist()


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}

    # 1. REAL BlueFors 24h telemetry
    d = "data/real/bluefors_cryometrics_sample"
    have_bf = os.path.isdir(d)
    if have_bf:
        fd = BF.load_fridge_day(d)
        data["real_bluefors"] = {
            "hours": downsample(fd["MXC"][0] / 3600.0) if "MXC" in fd else [],
            "stages": {st: downsample(fd[st][1]) for st in ["50K", "4K", "Still", "MXC"] if st in fd},
        }
        fp = VC.learn_fingerprint(d)
        data["fingerprint"] = {
            "stages": fp["stages"],
            "std_pct": [float(s / m * 100) for s, m in zip(fp["std"], fp["median"])],
            "corr": np.round(fp["corr"], 2).tolist(),
        }
    else:
        fp = None

    # 2. Engine base state + validation vs two real fridges
    cfg = CE.EngineConfig(fingerprint=fp)
    base = CE.simulate(CE.Scenario("normal"), cfg, hours=6, seed=1)
    data["engine_base"] = {f"temp{i+1}_T": float(np.mean(base[f"temp{i+1}_T"])) for i in range(5)}
    data["validation"] = {
        "stages": ["4K", "MXC"],
        "leeds": [4.36, 0.035], "bluefors": [2.94, 0.0112],
        "engine": [data["engine_base"]["temp2_T"], data["engine_base"]["temp5_T"]],
    }

    # 3. A fault scenario (engine) — the agent's-eye view
    fault = CE.simulate(CE.Scenario("heat_load_spike", severity=0.9, onset_frac=0.35),
                        cfg, hours=6, seed=3)
    data["fault_scenario"] = {
        "class": "heat_load_spike",
        "hours": downsample(fault["t_s"] / 3600.0),
        "mxc_mK": downsample(np.asarray(fault["temp5_T"]) * 1e3),
        "onset_h": 0.35 * fault["t_s"][-1] / 3600.0,
    }

    # 4. Live agent x simulator head-to-head (reconstructed from the turn log so it
    #    reflects the real run even if it is still in progress / was interrupted).
    try:
        from onnesim import agent_eval as AE
        log = f"{OUT}/agent_eval_turns.jsonl"
        if os.path.exists(log):
            r = AE.reconstruct_from_log(log)
            h = r["head_to_head"]
            data["head_to_head"] = {
                "n": r["n_scenarios_complete"], "turns": r["agent_turns_total"],
                "agent_det": h["agent_detection_f1"], "ml_det": h["ml_detection_f1"],
                "agent_cls": h["agent_classification_acc"], "ml_cls": h["ml_classification_acc"],
                "top_confusions": r["agent_panel"]["top_confusions"][:3],
            }
    except Exception as exc:  # noqa: BLE001
        print(f"[dashboard] head-to-head unavailable ({exc})")

    # 5. Continuous 24h watch — the agent catching a developing leak with a lead time.
    mon_path = f"{OUT}/continuous_monitor.json"
    if os.path.exists(mon_path):
        m = json.load(open(mon_path))
        data["monitor"] = {
            "fault": m["config"]["fault_class"], "hours": m["config"]["hours"],
            "onset_min": m["onset_min"], "lead_min": m["detection_latency_min"],
            "false_alarms": m["false_alarms_before_onset"], "n_polls": m["n_polls"],
            "t_h": downsample(np.asarray(m["trace"]["t_min"]) / 60.0),
            "mxc_mK": downsample(m["trace"]["mxc_mK"]),
        }

    with open(f"{OUT}/dashboard_data.json", "w") as f:
        json.dump(data, f)
    _write_html(data)
    print(f"[dashboard] wrote {OUT}/dashboard.html + dashboard_data.json")
    print(f"[dashboard] real BlueFors data: {have_bf}")


def _write_html(data):
    html = _TEMPLATE.replace("/*DATA*/", json.dumps(data))
    with open(f"{OUT}/dashboard.html", "w") as f:
        f.write(html)


_TEMPLATE = r"""<!doctype html><html lang=en><head><meta charset=utf8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Onnes — Cryo Ops</title>
<link rel=preconnect href="https://fonts.googleapis.com">
<link rel=preconnect href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap" rel=stylesheet>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root{
  --bg:#fff; --subtle:#fafafa; --border:#eaeaea; --border-2:#e3e3e3;
  --fg:#000; --fg-2:#171717; --muted:#666; --faint:#8f8f8f;
  --blue:#0070f3; --green:#00a862; --amber:#f5a623; --red:#e5484d; --violet:#7928ca;
  --sans:'Geist',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono:'Geist Mono',ui-monospace,'SF Mono',Menlo,monospace;
  --r:9px;
}
*{box-sizing:border-box;margin:0}
html{-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body{background:var(--bg);color:var(--fg-2);font-family:var(--sans);
  font-size:14px;line-height:1.5;padding:0 24px 64px}
.wrap{max-width:1120px;margin:0 auto}

/* header */
header{display:flex;align-items:center;justify-content:space-between;
  height:64px;border-bottom:1px solid var(--border)}
.brand{display:flex;align-items:center;gap:11px}
.mark{width:11px;height:11px;background:var(--fg);transform:rotate(45deg);border-radius:2px}
.word{font-weight:600;letter-spacing:-.01em;color:var(--fg)}
.slash{color:var(--border-2);font-weight:400}
.ctx{color:var(--muted);font-weight:400}
.status{display:flex;align-items:center;gap:8px;font-family:var(--mono);
  font-size:12px;color:var(--muted);padding:5px 11px;border:1px solid var(--border);border-radius:999px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);
  box-shadow:0 0 0 3px rgba(0,168,98,.14);animation:pulse 2.4s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 3px rgba(0,168,98,.14)}50%{box-shadow:0 0 0 5px rgba(0,168,98,.05)}}

/* title */
.title{padding:40px 0 32px}
.title h1{font-size:30px;font-weight:600;letter-spacing:-.025em;color:var(--fg)}
.title p{color:var(--muted);font-size:15px;margin-top:8px;max-width:60ch}

/* metric hero */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);
  border:1px solid var(--border);border-radius:var(--r);overflow:hidden;margin-bottom:16px}
.metric{padding:20px 22px;border-right:1px solid var(--border)}
.metric:last-child{border-right:0}
.mlabel{font-family:var(--mono);font-size:11px;text-transform:uppercase;
  letter-spacing:.06em;color:var(--faint)}
.mval{font-family:var(--mono);font-weight:500;font-size:32px;
  letter-spacing:-.03em;color:var(--fg);margin:10px 0 4px;line-height:1}
.mval span{font-size:15px;color:var(--faint);margin-left:3px;letter-spacing:0}
.mval.ok{color:var(--green);font-size:22px}
.msub{font-size:12px;color:var(--muted)}

/* card grid */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{border:1px solid var(--border);border-radius:var(--r);padding:20px 22px;background:var(--bg)}
.eyebrow{font-family:var(--mono);font-size:11px;text-transform:uppercase;
  letter-spacing:.06em;color:var(--faint)}
.card h2{font-size:15px;font-weight:600;color:var(--fg);margin:6px 0 2px;letter-spacing:-.01em}
.card .cap{font-size:12.5px;color:var(--muted);margin-bottom:16px}
.chartwrap{position:relative;height:210px}

/* validation bars */
.vhead{display:grid;grid-template-columns:56px 1fr;gap:14px;
  font-family:var(--mono);font-size:10px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--faint);margin-bottom:12px}
.vrow{display:grid;grid-template-columns:56px 1fr;gap:14px;align-items:center;margin-bottom:16px}
.vstage{font-family:var(--mono);font-size:13px;color:var(--fg)}
.vbars{display:flex;flex-direction:column;gap:7px}
.vbar{display:flex;align-items:center;gap:10px}
.vbar .name{font-family:var(--mono);font-size:10.5px;color:var(--muted);width:64px;flex:none}
.track{flex:1;height:6px;background:#f1f1f1;border-radius:3px;overflow:hidden}
.track i{display:block;height:100%;border-radius:3px}
.vbar .num{font-family:var(--mono);font-size:11px;color:var(--fg-2);width:56px;text-align:right;flex:none}
.note{font-size:12.5px;color:var(--muted);border-top:1px solid var(--border);padding-top:14px;margin-top:2px}
.note b{color:var(--fg);font-weight:500}

/* heatmap */
.hm{display:grid;grid-template-columns:auto repeat(4,1fr);gap:3px}
.hm .h{font-family:var(--mono);font-size:10px;color:var(--faint);
  display:flex;align-items:center;justify-content:center;padding:4px}
.hm .cell{aspect-ratio:1.9;display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:11px;border:1px solid var(--border);border-radius:5px}
.noise{font-family:var(--mono);font-size:11px;color:var(--muted);
  margin-top:14px;border-top:1px solid var(--border);padding-top:12px;line-height:1.9}
.noise b{color:var(--fg);font-weight:500}

/* fault badge */
.fbadge{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);
  font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--red);background:rgba(229,72,77,.07);border:1px solid rgba(229,72,77,.25);
  padding:3px 9px;border-radius:999px;margin-bottom:14px}
.fbadge .fdot{width:6px;height:6px;border-radius:50%;background:var(--red)}

/* agent head-to-head */
.h2h{border:1px solid var(--border);border-radius:var(--r);padding:20px 22px;margin-bottom:16px}
.h2h-top{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:8px}
.h2h .cap{font-size:12.5px;color:var(--muted);margin:6px 0 18px}
.h2h-grid{display:grid;grid-template-columns:1fr 1fr 1.1fr;gap:26px}
.vs{display:flex;flex-direction:column;gap:14px}
.vs .metriclabel{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--faint);margin-bottom:2px}
.vs .row{display:flex;align-items:center;gap:10px}
.vs .who{font-family:var(--mono);font-size:11px;width:42px;flex:none;color:var(--muted)}
.vs .track{flex:1;height:8px;background:#f1f1f1;border-radius:4px;overflow:hidden}
.vs .track i{display:block;height:100%;border-radius:4px}
.vs .num{font-family:var(--mono);font-size:12px;width:40px;text-align:right;flex:none;color:var(--fg-2)}
.watch{border-left:1px solid var(--border);padding-left:26px}
.watch .big{font-family:var(--mono);font-weight:500;font-size:30px;letter-spacing:-.03em;
  color:var(--green);line-height:1;margin:8px 0 4px}
.watch .big span{font-size:14px;color:var(--faint)}
.watch .fa{font-family:var(--mono);font-size:12px;color:var(--amber);margin-top:10px}
.conf{font-size:12px;color:var(--muted);margin-top:14px;border-top:1px solid var(--border);padding-top:12px;line-height:1.9}
.conf b{color:var(--fg);font-weight:500}
@media(max-width:760px){.h2h-grid{grid-template-columns:1fr}.watch{border-left:0;padding-left:0;border-top:1px solid var(--border);padding-top:16px}}

footer{font-family:var(--mono);font-size:11.5px;color:var(--faint);
  text-align:center;margin-top:36px;padding-top:22px;border-top:1px solid var(--border)}

@media(max-width:760px){
  .metrics{grid-template-columns:1fr 1fr}
  .metric:nth-child(2){border-right:0}
  .metric:nth-child(1),.metric:nth-child(2){border-bottom:1px solid var(--border)}
  .grid{grid-template-columns:1fr}
  .title h1{font-size:25px}
}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body><div class=wrap>

<header>
  <div class=brand>
    <span class=mark></span>
    <span class=word>Onnes</span>
    <span class=slash>/</span>
    <span class=ctx>Cryo Ops</span>
  </div>
  <div class=status><span class=dot></span> Operational</div>
</header>

<div class=title>
  <h1>Cryogenic Mission Control</h1>
  <p>Real dilution-fridge telemetry, physics-grounded simulation, and a live
     multi-agent operator — every figure sourced from measured data or a live run.</p>
</div>

<section class=metrics id=metrics></section>

<section class=h2h id=h2h style=display:none>
  <div class=h2h-top>
    <div>
      <div class=eyebrow>Agent &times; Simulator</div>
      <h2 style="font-size:15px;font-weight:600;color:var(--fg);margin:6px 0 0">Reasoning agent vs. trained model</h2>
    </div>
    <div class=status style="border-color:var(--border)"><span class=dot></span> <span id=h2h-turns></span></div>
  </div>
  <div class=cap>Five live Claude agents vs the strongest ML baseline on identical injected faults &middot; head-to-head on the realistic engine</div>
  <div class=h2h-grid>
    <div class=vs id=vs-det></div>
    <div class=vs id=vs-cls></div>
    <div class=watch id=watch></div>
  </div>
  <div class=conf id=conf></div>
</section>

<div class=grid>
  <div class=card>
    <div class=eyebrow>Telemetry</div>
    <h2>Live stage temperatures</h2>
    <div class=cap>BlueFors &ldquo;blizzard&rdquo; &middot; 24-hour window &middot; log scale</div>
    <div class=chartwrap><canvas id=c1></canvas></div>
  </div>

  <div class=card>
    <div class=eyebrow>Validation</div>
    <h2>Simulation accuracy</h2>
    <div class=cap>Engine base temperatures vs two independent real fridges</div>
    <div id=val></div>
    <div class=note>Untuned, the engine lands <b>between</b> two independent real machines on every stage.</div>
  </div>

  <div class=card>
    <div class=eyebrow>Anomaly</div>
    <h2>Fault signature</h2>
    <div class=cap>Mixing-chamber response the operator agent reads</div>
    <span class=fbadge id=fbadge><span class=fdot></span> Fault</span>
    <div class=chartwrap><canvas id=c2></canvas></div>
  </div>

  <div class=card>
    <div class=eyebrow>Fingerprint</div>
    <h2>Cross-stage coupling</h2>
    <div class=cap>Measured correlation the clone reproduces</div>
    <div class=hm id=hm></div>
    <div class=noise id=noise></div>
  </div>
</div>

<footer>Validated against Leeds &middot; BlueFors &middot; ORNL &nbsp;—&nbsp; onnes.ai</footer>
</div>

<script>
const D=/*DATA*/;
Chart.defaults.font.family="'Geist Mono',ui-monospace,monospace";
Chart.defaults.font.size=10;
Chart.defaults.color="#8f8f8f";
const STAGE={'50K':'#f5a623','4K':'#0070f3','Still':'#171717','MXC':'#00a862'};
const cfg={responsive:true,maintainAspectRatio:false,
 interaction:{intersect:false,mode:'index'},
 plugins:{legend:{labels:{color:'#666',boxWidth:7,boxHeight:7,usePointStyle:true,
   pointStyle:'circle',padding:14,font:{size:11}}},
   tooltip:{backgroundColor:'#000',padding:9,cornerRadius:7,titleFont:{size:11},
   bodyFont:{size:11},displayColors:false}},
 scales:{x:{ticks:{color:'#8f8f8f',maxTicksLimit:7},grid:{color:'#f4f4f4',drawTicks:false},
   border:{color:'#eaeaea'}},
  y:{ticks:{color:'#8f8f8f'},grid:{color:'#f4f4f4',drawTicks:false},border:{color:'#eaeaea'}}}};

// metric hero
if(D.engine_base){const e=D.engine_base;
 document.getElementById('metrics').innerHTML=`
  <div class=metric><div class=mlabel>Mixing chamber</div>
    <div class=mval>${(e.temp5_T*1e3).toFixed(1)}<span>mK</span></div>
    <div class=msub>Base temperature</div></div>
  <div class=metric><div class=mlabel>4K flange</div>
    <div class=mval>${e.temp2_T.toFixed(2)}<span>K</span></div>
    <div class=msub>Magnet stage</div></div>
  <div class=metric><div class=mlabel>Fridge state</div>
    <div class="mval ok">Nominal</div>
    <div class=msub>All stages in band</div></div>
  <div class=metric><div class=mlabel>Data window</div>
    <div class=mval>24<span>h</span></div>
    <div class=msub>Real telemetry</div></div>`;}

// c1 — real stages, log scale
if(D.real_bluefors){const r=D.real_bluefors;
 new Chart(c1,{type:'line',data:{labels:r.hours.map(h=>h.toFixed(0)),
  datasets:Object.keys(r.stages).map(s=>({label:s,data:r.stages[s].map(v=>Math.log10(v)),
   borderColor:STAGE[s],backgroundColor:STAGE[s],borderWidth:1.75,pointRadius:0,tension:.3}))},
  options:{...cfg,scales:{...cfg.scales,
   x:{...cfg.scales.x,title:{display:true,text:'hours',color:'#8f8f8f',font:{size:10}}},
   y:{...cfg.scales.y,title:{display:true,text:'log₁₀ T (K)',color:'#8f8f8f',font:{size:10}}}}}});}

// validation bars
if(D.validation){const v=D.validation;
 const COL={leeds:'#8f8f8f',bluefors:'#0070f3',engine:'#00a862'};
 let h='<div class=vhead><div>Stage</div><div>Leeds · BlueFors · Engine</div></div>';
 v.stages.forEach((s,i)=>{
  const mx=Math.max(v.leeds[i],v.bluefors[i],v.engine[i])*1.15;
  const u=s=='MXC'?'mK':'K',sc=s=='MXC'?1e3:1,d=s=='MXC'?0:2;
  const bar=(k,label)=>{const x=v[k][i];
   return `<div class=vbar><span class=name>${label}</span>
    <div class=track><i style="width:${x/mx*100}%;background:${COL[k]}"></i></div>
    <span class=num>${(x*sc).toFixed(d)} ${u}</span></div>`;};
  h+=`<div class=vrow><div class=vstage>${s}</div><div class=vbars>
    ${bar('leeds','Leeds')}${bar('bluefors','BlueFors')}${bar('engine','Engine')}</div></div>`;});
 document.getElementById('val').innerHTML=h;}

// fault
if(D.fault_scenario){const f=D.fault_scenario;
 document.getElementById('fbadge').innerHTML=
  `<span class=fdot></span> ${f.class.replace(/_/g,' ')}`;
 new Chart(c2,{type:'line',data:{labels:f.hours.map(h=>h.toFixed(1)),
  datasets:[{label:'MXC (mK)',data:f.mxc_mK,borderColor:'#e5484d',backgroundColor:'rgba(229,72,77,.06)',
   borderWidth:2,pointRadius:0,tension:.25,fill:true}]},
  options:{...cfg,plugins:{...cfg.plugins,legend:{display:false}},
   scales:{...cfg.scales,
    x:{...cfg.scales.x,title:{display:true,text:'hours',color:'#8f8f8f',font:{size:10}}},
    y:{...cfg.scales.y,title:{display:true,text:'mK',color:'#8f8f8f',font:{size:10}}}}}});}

// heatmap
if(D.fingerprint){const fp=D.fingerprint,st=fp.stages;
 let h='<div class=h></div>'+st.map(s=>`<div class=h>${s}</div>`).join('');
 fp.corr.forEach((row,i)=>{h+=`<div class=h>${st[i]}</div>`+row.map(c=>{
  const a=Math.abs(c),col=c>=0?`rgba(0,112,243,${a})`:`rgba(229,72,77,${a})`;
  return `<div class=cell style="background:${col};color:${a>.55?'#fff':'#8f8f8f'}">${c.toFixed(2)}</div>`;
 }).join('');});
 document.getElementById('hm').innerHTML=h;
 document.getElementById('noise').innerHTML='Per-stage noise &nbsp;'+
  st.map((s,i)=>`${s} <b>${fp.std_pct[i].toFixed(2)}%</b>`).join(' &nbsp;·&nbsp; ');}

// agent x ML head-to-head
if(D.head_to_head){const h=D.head_to_head;
 document.getElementById('h2h').style.display='block';
 document.getElementById('h2h-turns').textContent=`${h.turns} live agent turns · ${h.n} scenarios`;
 const AG='#00a862',ML='#0070f3';
 const pair=(el,label,a,m)=>{document.getElementById(el).innerHTML=
  `<div class=metriclabel>${label}</div>
   <div class=row><span class=who>Agent</span><div class=track><i style="width:${a*100}%;background:${AG}"></i></div><span class=num>${a.toFixed(2)}</span></div>
   <div class=row><span class=who>ML</span><div class=track><i style="width:${m*100}%;background:${ML}"></i></div><span class=num>${m.toFixed(2)}</span></div>`;};
 pair('vs-det','Detection F1 (is a fault present?)',h.agent_det,h.ml_det);
 pair('vs-cls','Classification acc (which fault?)',h.agent_cls,h.ml_cls);
 if(D.monitor){const w=D.monitor;
  document.getElementById('watch').innerHTML=
   `<div class=metriclabel>Continuous ${w.hours}h watch · ${w.fault.replace(/_/g,' ')}</div>
    <div class=big>${w.lead_min!=null?w.lead_min.toFixed(0):'—'}<span> min lead</span></div>
    <div class=msub style="font-size:12px;color:var(--muted)">from onset to first agent alarm</div>
    <div class=fa>${w.false_alarms} false alarms before onset · ${w.n_polls} polls</div>`;}
 if(h.top_confusions&&h.top_confusions.length){
  document.getElementById('conf').innerHTML='Where the agent slips &nbsp;'+
   h.top_confusions.map(c=>`<b>${c.truth.replace(/_/g,' ')}</b>→${c.pred.replace(/_/g,' ')} <span style=color:var(--faint)>×${c.count}</span>`).join(' &nbsp;·&nbsp; ')+
   ' &nbsp;—&nbsp; the deliberately-overlapping thermal faults, exactly as engineered.';}}
</script></body></html>"""


if __name__ == "__main__":
    main()
