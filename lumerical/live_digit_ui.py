"""Presentation UI for one handwritten digit through the 16-channel layer.

Layout follows the demo spec: input (left), 16-channel matrix comparison
(center), classification (right).  The default mode is the saved optical
matrix so an 8 GB demo laptop does not need to start Lumerical.  A separate
history view replays a saved live run.

Transport is stdlib HTTP + JSON; the recognition itself is in live_digit.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumerical.live_digit import DEFAULT_OUT, recognize_pixels  # noqa: E402

HISTORY_RECORD = ROOT / "progress/07_lumerical/live_digit_proof/session_5a8b4f9112.json"
TOPOLOGY_RECORD = ROOT / "progress/08_live_digit/model-topology.json"

HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>手写数字识别中的光路矩阵计算</title>
<style>
:root { color-scheme: light; --ink:#17202a; --muted:#66727c; --line:#d9e0e6; --blue:#1769aa; --orange:#d97706; }
* { box-sizing: border-box; }
body { margin:0; background:#f3f5f7; color:var(--ink); font-family:"Segoe UI",system-ui,sans-serif; }
header { max-width:1500px; margin:0 auto; padding:18px 22px 8px; display:flex; align-items:flex-end; justify-content:space-between; gap:16px; }
h1 { font-size:25px; margin:0; letter-spacing:.01em; }
.subtitle { color:var(--muted); font-size:13px; margin-top:4px; }
#sourceBadge { border-radius:999px; padding:7px 12px; background:#e8edf1; color:#45525c; font-size:12px; font-weight:700; white-space:nowrap; }
#sourceBadge.live { background:#d93025; color:#fff; }
#sourceBadge.cached { background:#e9eef2; color:#45525c; }
#sourceBadge.digital { background:#e9eef2; color:#45525c; }
main { max-width:1500px; margin:0 auto; padding:10px 22px 26px; display:grid; grid-template-columns:minmax(250px,1fr) minmax(430px,2fr) minmax(230px,.95fr); gap:16px; align-items:stretch; }
.card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:15px; box-shadow:0 2px 10px #00000008; min-width:0; height:100%; }
.classifyCard { display:flex; flex-direction:column; justify-content:center; }
.card h2 { font-size:15px; margin:0 0 10px; color:#35424c; }
canvas.pad { width:100%; max-width:200px; aspect-ratio:1; border:1px solid #b9c4cc; border-radius:10px; touch-action:none; cursor:crosshair; display:block; margin:0 auto; background:#fff; }
.row { display:flex; gap:8px; flex-wrap:wrap; margin-top:7px; align-items:center; }
button { border:0; border-radius:8px; padding:9px 12px; font-size:13px; cursor:pointer; background:#e9eef2; color:#23313c; }
button.primary { background:var(--blue); color:#fff; }
button:disabled { opacity:.55; cursor:wait; }
label { font-size:12.5px; color:#35424c; display:flex; gap:6px; align-items:center; }
#status { color:var(--muted); font-size:12.5px; margin-top:7px; min-height:18px; }
#previewWrap { margin-top:8px; text-align:center; }
#preview { width:80px; height:80px; image-rendering:pixelated; border:1px solid var(--line); border-radius:8px; background:#fff; }
.caption { color:var(--muted); font-size:11.5px; margin-top:4px; }
#chart { width:100%; height:260px; display:block; border:1px solid #e4eaee; border-radius:10px; background:#fff; }
.bottom { max-width:1500px; margin:0 auto; padding:0 22px 26px; display:grid; grid-template-columns:minmax(360px,1fr) minmax(420px,1.45fr); gap:16px; align-items:start; }
.bottom[hidden] { display:none; }
.topoCard { background:#102731; border-color:#102731; color:#eaf4f7; padding:14px; display:flex; flex-direction:column; }
.topoCard h2, .topoCard .caption { color:#d8e8ee; }
.topoCard .caption { color:#9fb8c4; }
#topoSvg { width:100%; height:auto; display:block; margin:auto 0; }
#historyTopoSvg { width:100%; height:210px; display:block; background:#102731; border-radius:8px; margin:8px 0; }
.mziCard svg { width:100%; height:auto; background:#fbfcfd; border:1px solid #e4eaee; border-radius:8px; margin-top:8px; }
#mziInfo { margin-top:8px; font-size:12px; color:#45525c; }
.topoNote { font-size:11.5px; color:#9fb8c4; margin-top:8px; }
.matrixHead { display:flex; justify-content:space-between; gap:10px; align-items:baseline; }
#sourceLabel { font-size:12.5px; color:var(--muted); }
#matrixError { margin:8px 0 0; font-size:13px; font-weight:700; }
#matrixError.none { color:var(--muted); font-weight:400; }
.legend { font-size:12px; color:var(--muted); margin:6px 0 8px; }
.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 4px 0 0; }
.big { font-size:64px; line-height:1.05; font-weight:800; text-align:center; margin:18px 0 4px; }
.classHint { text-align:center; color:var(--muted); font-size:12.5px; }
.predGrid { display:grid; grid-template-columns:1fr 1fr; gap:9px; margin-top:18px; }
.predBox { background:#f7f9fa; border-radius:9px; padding:9px; text-align:center; }
.predBox span { display:block; color:var(--muted); font-size:11px; margin-bottom:4px; }
.predBox b { font-size:20px; }
#agreement { text-align:center; margin-top:12px; font-size:13px; }
details { margin-top:12px; }
summary { cursor:pointer; color:#35424c; font-size:13px; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; background:#f7f9fa; border-radius:8px; padding:9px; font:11.5px/1.5 Consolas,monospace; color:#3e4b55; }
.historyWrap { max-width:1500px; margin:0 auto; padding:10px 22px 26px; }
.historyTop { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:12px; }
.historyGrid { display:grid; grid-template-columns:minmax(240px,1fr) minmax(430px,2fr) minmax(230px,.95fr); gap:16px; align-items:stretch; }
#historyView[hidden], main[hidden] { display:none; }
@media(max-width:1050px) { main, .historyGrid, .bottom { grid-template-columns:1fr; } header { align-items:flex-start; flex-direction:column; } canvas.pad { max-width:280px; } }
</style>
</head>
<body>
<header>
  <div>
    <h1>手写数字识别中的光路矩阵计算</h1>
    <div class="subtitle">16 个特征 → 16×16 矩阵乘法 → 16 个输出 → 数字分类。软件基线、已保存光路矩阵和现场 Lumerical 三种来源严格区分。</div>
  </div>
  <div id="sourceBadge">等待识别</div>
</header>

<main id="liveView">
  <section class="card">
    <h2>输入</h2>
    <canvas id="pad" class="pad" width="300" height="300"></canvas>
    <div class="row">
      <button id="clear">清空</button>
      <button id="upload">上传</button>
      <input id="file" type="file" accept="image/*" hidden>
    </div>
    <div class="row" id="modes">
      <label><input type="radio" name="mode" value="cached"> 使用已保存光路矩阵</label>
      <label><input type="radio" name="mode" value="live" checked> 本次运行 Lumerical</label>
      <label><input type="radio" name="mode" value="digital"> 仅软件计算</label>
    </div>
    <div class="row"><button id="run" class="primary">识别</button><button id="historyBtn">查看已保存的仿真</button></div>
    <div id="status">等待识别。</div>
    <div id="previewWrap"><div class="caption">模型实际输入 28×28</div><canvas id="preview" width="140" height="140"></canvas></div>
  </section>

  <section class="card topoCard">
    <h2>光路矩阵层</h2>
    <div class="caption">输入混合 V† → 幅度缩放 Σ → 输出混合 U</div>
    <svg id="topoSvg" viewBox="0 0 1000 300" role="img" aria-label="固定 16 通道光路拓扑"></svg>
    <div class="topoNote">240 个 MZI · 16 个衰减器；中央结构固定，不随输入变化。绿色块连接相邻两条通道。</div>
  </section>

  <section class="card classifyCard">
    <h2>分类</h2>
    <div id="prediction" class="big">—</div>
    <div class="classHint">程序根据这 16 个输出完成分类</div>
    <div class="predGrid">
      <div class="predBox"><span>软件直接计算</span><b id="digitalPred">—</b></div>
      <div class="predBox"><span>光路输出</span><b id="opticalPred">—</b></div>
    </div>
    <div id="agreement" class="caption">—</div>
    <details id="detailsMain"><summary>查看运行细节</summary><div id="detailMain" class="caption">尚未识别。</div></details>
  </section>
</main>

<section class="bottom" id="liveBottom">
  <section class="card mziCard">
    <h2>放大一个 MZI</h2>
    <div class="row" style="margin-top:0"><button id="expandMzi">展开第一个 MZI</button><span class="caption">或点击中央绿色块切换</span></div>
    <svg id="mziSvg" viewBox="0 0 520 160" role="img" aria-label="单个 MZI 放大图"></svg>
    <div id="mziInfo">尚未选择 MZI。</div>
  </section>
  <section class="card">
    <h2>16 路线性输出</h2>
    <div class="matrixHead"><span id="sourceLabel">等待识别</span><span class="caption">同一刻度 / 保留正负号 / 零线</span></div>
    <div class="legend"><span class="sw" style="background:var(--blue)"></span>软件直接计算 <span class="sw" style="background:var(--orange);margin-left:12px"></span>光路（缓存或本次仿真）</div>
    <canvas id="chart" width="640" height="260"></canvas>
    <div id="matrixError" class="none">16 路矩阵计算相对误差：—</div>
  </section>
</section>

<section id="historyView" class="historyWrap" hidden>
  <div class="historyTop">
    <button id="back">返回画板</button>
    <b id="historyTitle">历史回放 · 已完成的 Lumerical 仿真</b>
    <span id="historyRecord" class="caption"></span>
  </div>
  <div class="historyGrid">
    <section class="card">
      <h2>当时的输入</h2>
      <canvas id="historyImage" width="280" height="280" style="width:180px;height:180px;image-rendering:pixelated;border:1px solid var(--line);border-radius:8px;background:#fff"></canvas>
      <div class="caption" style="margin-top:8px">记录中的 28×28 输入；不是本次画板。</div>
    </section>
    <section class="card">
      <h2>当时的光路与 16 路对照</h2>
      <div id="historySource" class="caption"></div>
      <svg id="historyTopoSvg" viewBox="0 0 1000 300" aria-label="历史回放的固定光路拓扑"></svg>
      <div class="legend"><span class="sw" style="background:var(--blue)"></span>当时的软件计算 <span class="sw" style="background:var(--orange);margin-left:12px"></span>当时的电路仿真</div>
      <canvas id="historyChart" width="640" height="300" style="width:100%;height:300px;border:1px solid #e4eaee;border-radius:10px"></canvas>
      <div id="historyError" class="caption"></div>
      <details><summary>查看历史运行细节</summary><div id="historyDetail" class="caption"></div></details>
    </section>
    <section class="card classifyCard">
      <h2>当时的分类</h2>
      <div id="historyPrediction" class="big">—</div>
      <div class="predGrid">
        <div class="predBox"><span>软件直接计算</span><b id="historyDigital">—</b></div>
        <div class="predBox"><span>电路仿真</span><b id="historyOptical">—</b></div>
      </div>
      <div id="historyAgreement" class="caption"></div>
      <div class="caption" style="margin-top:14px">这一段约 97 秒属于历史运行，不是本次画板的耗时。</div>
    </section>
  </div>
</section>

<script>
const sourceBadge=document.getElementById('sourceBadge'), statusEl=document.getElementById('status'), detailsMain=document.getElementById('detailMain');
const pad=document.getElementById('pad'), ctx=pad.getContext('2d'), preview=document.getElementById('preview'), pctx=preview.getContext('2d');
let drawing=false, hasInk=false, requestSeq=0, busy=false, lastResult=null, waitTimer=null, modeTouched=false;
document.querySelectorAll('input[name="mode"]').forEach(el=>el.addEventListener('change',()=>{modeTouched=true;}));

function white(){ ctx.fillStyle='#fff'; ctx.fillRect(0,0,300,300); ctx.strokeStyle='#111'; ctx.lineWidth=19; ctx.lineCap='round'; ctx.lineJoin='round'; }
white();
function clearResult(){
  sourceBadge.textContent='等待识别'; sourceBadge.className='';
  document.getElementById('sourceLabel').textContent='等待识别';
  document.getElementById('matrixError').textContent='16 路矩阵计算相对误差：—';
  document.getElementById('matrixError').className='none';
  document.getElementById('prediction').textContent='—';
  document.getElementById('digitalPred').textContent='—';
  document.getElementById('opticalPred').textContent='—';
  document.getElementById('agreement').textContent='—';
  detailsMain.textContent='尚未识别。';
  const c=document.getElementById('chart'), g=c.getContext('2d'); g.clearRect(0,0,c.width,c.height);
  const p=pctx; p.clearRect(0,0,140,140); p.fillStyle='#fff'; p.fillRect(0,0,140,140);
}
function invalidate(){ requestSeq++; lastResult=null; clearResult(); statusEl.textContent='等待识别。'; }

const SVGNS='http://www.w3.org/2000/svg';
let topology=null, selectedGate=null;
function svgEl(tag, attrs, parent){ const e=document.createElementNS(SVGNS,tag); for(const [k,v] of Object.entries(attrs||{})) e.setAttribute(k,String(v)); parent.appendChild(e); return e; }
function placeMesh(ops, left, right){
  const last=Array(16).fill(-1);
  const gates=ops.map(([m,theta,phi])=>{
    if(!Number.isInteger(m)||m<0||m>=15) throw new Error('MZI 通道编号越界');
    const column=1+Math.max(last[m],last[m+1]); last[m]=last[m+1]=column;
    return {m,theta,phi,column};
  });
  const columns=1+Math.max(...last);
  return gates.map(g=>({...g,x:left+(g.column+.5)*(right-left)/columns,y:70+12*g.m}));
}
function renderTopology(id, topo, interactive){
  const svg=document.getElementById(id); svg.innerHTML='';
  const left=90,right=410,left2=540,right2=860, att=475;
  svgEl('text',{x:(left+right)/2,y:26,fill:'#d8e8ee','font-size':15,'text-anchor':'middle'},svg).textContent='输入混合 V†';
  svgEl('text',{x:att,y:26,fill:'#d8e8ee','font-size':15,'text-anchor':'middle'},svg).textContent='幅度缩放 Σ';
  svgEl('text',{x:(left2+right2)/2,y:26,fill:'#d8e8ee','font-size':15,'text-anchor':'middle'},svg).textContent='输出混合 U';
  for(let i=0;i<16;i++){ const y=70+12*i; svgEl('line',{x1:55,y1:y,x2:960,y2:y,stroke:'#496875','stroke-width':1.2},svg); }
  const gates=placeMesh(topo.input_mesh.ops,left,right).concat(placeMesh(topo.output_mesh.ops,left2,right2));
  for(const g of gates){
    const r=svgEl('rect',{x:g.x-4,y:g.y-2,width:8,height:16,rx:2,fill:'#39c6b8',style:'cursor:pointer'},svg);
    const tip=svgEl('title',{},r);
    tip.textContent=`${g.x<att?'输入':'输出'}网格 MZI：通道 ${g.m+1}/${g.m+2}，θ=${Number(g.theta).toFixed(4)}，φ=${Number(g.phi).toFixed(4)} rad`;
    if(interactive) r.addEventListener('click',()=>{ selectedGate=g; renderMzi(g); });
  }
  for(let i=0;i<topo.amplitude_transmissions.length;i++){
    const y=70+12*i;
    const a=svgEl('rect',{x:att-10,y:y-4,width:20,height:8,rx:2,fill:'#e0a42a'},svg);
    const tip=svgEl('title',{},a); tip.textContent=`通道 ${i+1} 幅度透过率 ${Number(topo.amplitude_transmissions[i]).toFixed(5)}`;
  }
  for(let i=0;i<16;i++){
    const y=70+12*i;
    const c1=svgEl('circle',{cx:430,cy:y,r:2.2,fill:'#9fd8e6'},svg);
    const c2=svgEl('circle',{cx:890,cy:y,r:2.2,fill:'#9fd8e6'},svg);
    const t1=svgEl('title',{},c1); t1.textContent=`输入网格输出相位 ${Number(topo.input_mesh.output_phases[i]).toFixed(4)} rad`;
    const t2=svgEl('title',{},c2); t2.textContent=`输出网格输出相位 ${Number(topo.output_mesh.output_phases[i]).toFixed(4)} rad`;
  }
}
function renderMzi(g){
  if(!g)return;
  const svg=document.getElementById('mziSvg'); svg.innerHTML='';
  svgEl('line',{x1:10,y1:55,x2:510,y2:55,stroke:'#2c697a','stroke-width':2},svg);
  svgEl('line',{x1:10,y1:105,x2:510,y2:105,stroke:'#2c697a','stroke-width':2},svg);
  for(const x of [125,355]){
    svgEl('rect',{x:x-9,y:40,width:18,height:80,rx:4,fill:'#d9f1ef',stroke:'#2c9b92','stroke-width':2},svg);
    svgEl('line',{x1:x-9,y1:40,x2:x+9,y2:120,stroke:'#2c9b92','stroke-width':2},svg);
    svgEl('line',{x1:x+9,y1:40,x2:x-9,y2:120,stroke:'#2c9b92','stroke-width':2},svg);
  }
  svgEl('text',{x:125,y:145,fill:'#45525c','font-size':11,'text-anchor':'middle'},svg).textContent='分光';
  svgEl('text',{x:355,y:145,fill:'#45525c','font-size':11,'text-anchor':'middle'},svg).textContent='合束干涉';
  for(const [x,y,label,val] of [[60,44,'φ',g.phi],[240,44,'2θ',2*g.theta],[445,34,'−θ',-g.theta]]){
    svgEl('rect',{x:x-8,y:y-4,width:16,height:8,rx:3,fill:'#e0a42a'},svg);
    svgEl('text',{x:x,y:y-9,fill:'#9a5b00','font-size':11,'text-anchor':'middle'},svg).textContent=`${label}=${Number(val).toFixed(3)}`;
  }
  document.getElementById('mziInfo').textContent=`通道 ${g.m+1}/${g.m+2} · θ=${Number(g.theta).toFixed(4)} rad · φ=${Number(g.phi).toFixed(4)} rad`;
}
function pos(e){ const r=pad.getBoundingClientRect(); return {x:(e.clientX-r.left)*300/r.width, y:(e.clientY-r.top)*300/r.height}; }
pad.addEventListener('pointerdown', e=>{ drawing=true; hasInk=true; invalidate(); pad.setPointerCapture(e.pointerId); const p=pos(e); ctx.beginPath(); ctx.moveTo(p.x,p.y); });
pad.addEventListener('pointermove', e=>{ if(!drawing)return; const p=pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); });
pad.addEventListener('pointerup', ()=>{ drawing=false; });
document.getElementById('clear').onclick=()=>{ white(); hasInk=false; invalidate(); statusEl.textContent='已清空，等待识别。'; };
const file=document.getElementById('file');
document.getElementById('upload').onclick=()=>file.click();
file.onchange=()=>{ const f=file.files[0]; if(!f)return; const img=new Image(); img.onload=()=>{ white(); const s=Math.min(300/img.width,300/img.height); const w=img.width*s,h=img.height*s; ctx.fillStyle='#fff'; ctx.fillRect(0,0,300,300); ctx.drawImage(img,(300-w)/2,(300-h)/2,w,h); hasInk=true; invalidate(); statusEl.textContent='图片已载入，等待识别。'; }; img.src=URL.createObjectURL(f); };
function canvasPixels(){ const d=ctx.getImageData(0,0,300,300).data, out=[]; for(let i=0;i<d.length;i+=4) out.push((d[i]+d[i+1]+d[i+2])/3); return out; }
function drawImage28(target, img, scale){ const c=document.getElementById(target), g=c.getContext('2d'); g.fillStyle='#fff'; g.fillRect(0,0,c.width,c.height); const cell=c.width/28; for(let y=0;y<28;y++) for(let x=0;x<28;x++){ const v=255-img[y][x]; g.fillStyle=`rgb(${v},${v},${v})`; g.fillRect(x*cell,y*cell,cell,cell); } }
function drawChart(id, digital, optical){
  const c=document.getElementById(id), g=c.getContext('2d'); g.clearRect(0,0,c.width,c.height); g.fillStyle='#fff'; g.fillRect(0,0,c.width,c.height);
  const n=digital.length, left=34, right=10, top=16, bottom=28, mid=top+(c.height-top-bottom)/2;
  const vals=optical?digital.concat(optical):digital, max=Math.max(...vals.map(Math.abs),1e-9);
  const w=(c.width-left-right)/n, scale=(mid-top)/max;
  g.strokeStyle='#c8d1d8'; g.beginPath(); g.moveTo(left,mid); g.lineTo(c.width-right,mid); g.stroke();
  g.font='10px Consolas'; g.fillStyle='#66727c'; g.fillText('0',8,mid+3);
  for(let i=0;i<n;i++){ const x=left+i*w+w*.17; const bd=Math.abs(digital[i])*scale; g.fillStyle='#1769aa'; g.fillRect(x,digital[i]>=0?mid-bd:mid,w*.30,bd); if(optical){ const bo=Math.abs(optical[i])*scale; g.fillStyle='#d97706'; g.fillRect(x+w*.34,optical[i]>=0?mid-bo:mid,w*.30,bo);} g.fillStyle='#66727c'; g.fillText(String(i+1),x,c.height-8); }
}
function fmt(a){ return a.map(v=>Number(v).toExponential(2)).join(', '); }
function detailHtml(data){
  const o=data.optical;
  if(!o) return `<b>仅软件计算</b><br>没有调用 Lumerical；没有光路误差。`;
  const live=!!data.live_interconnect_called;
  let s=`<b>${live?'本次实际运行 Lumerical':'本次使用已保存 H（未启动求解器）'}</b><br>`;
  if(live) s+=`build ${Number(o.build_seconds).toFixed(2)} s / run ${Number(o.run_seconds).toFixed(2)} s / 总计 ${Number(data.elapsed_seconds).toFixed(2)} s<br>`;
  s+=`电路 ${o.circuit_size||16} 通道 / ${o.mzi_count||'未记录'} MZI；g=${Number(o.encoding_scale_g).toFixed(6)}，s=${Number(o.recovery_scale_s).toFixed(6)}<br>`;
  s+=`相对误差 ${Number(o.linear_relative_error).toExponential(3)}；复泄漏 ${Number(o.imaginary_leakage_ratio).toExponential(3)}<br>`;
  if(o.encoded_amplitude) s+=`编码幅度 |x|/g：${fmt(o.encoded_amplitude)}<br>`;
  if(o.encoded_phase_rad) s+=`编码相位：${fmt(o.encoded_phase_rad)}<br>`;
  if(o.raw_optical){ const r=o.raw_optical.real, im=o.raw_optical.imag; s+=`<br>原始复输出 y：<br>`+r.map((v,i)=>`${String(i+1).padStart(2,'0')} ${v.toFixed(6)} ${im[i]>=0?'+':''}${im[i].toFixed(6)}j`).join('<br>'); }
  if(data.session_file) s+=`<br><a href="/api/session?path=${encodeURIComponent(data.session_file)}">下载本次 session JSON</a>`;
  if(data.historical) s+=`<br><a href="/api/session?record=history">下载历史 session JSON</a>`;
  return s;
}
function sourceInfo(data){ if(data.mode==='live') return ['本次已运行 Lumerical 电路仿真','live']; if(data.mode==='cached') return ['本次使用已保存的光路矩阵','cached']; return ['本次仅运行软件计算','digital']; }
function renderMain(data){
  lastResult=data;
  drawImage28('preview',data.image_28);
  const [label,cls]=sourceInfo(data); sourceBadge.textContent=label; sourceBadge.className=cls; document.getElementById('sourceLabel').textContent=label;
  const o=data.optical;
  drawChart('chart',data.digital.linear,o?o.restored_linear.real:null);
  const e=document.getElementById('matrixError');
  if(o){ e.textContent=`16 路矩阵计算相对误差：${Number(o.linear_relative_error).toExponential(3)}`; e.className=''; }
  else { e.textContent='16 路矩阵计算相对误差：无对照'; e.className='none'; }
  document.getElementById('prediction').textContent=o?o.prediction:data.digital.prediction;
  document.getElementById('digitalPred').textContent=data.digital.prediction;
  document.getElementById('opticalPred').textContent=o?o.prediction:'—';
  document.getElementById('agreement').textContent=o?(data.prediction_agreement?'软件与光路预测一致':'软件与光路预测不一致'):'没有光路结果';
  detailsMain.innerHTML=detailHtml(data);
}
document.getElementById('run').onclick=async()=>{
  if(!hasInk){ statusEl.textContent='画板为空，请先写一个数字。'; return; }
  const seq=++requestSeq, snapshot=canvasPixels();
  const mode=document.querySelector('input[name="mode"]:checked').value;
  const btn=document.getElementById('run'); btn.disabled=true; busy=true;
  statusEl.textContent = mode==='live' ? '正在运行 Lumerical（等待 0 s）…' : (mode==='cached'?'正在应用已保存光路矩阵…':'正在计算软件基线…');
  const t0=performance.now();
  if(mode==='live') waitTimer=setInterval(()=>{ statusEl.textContent=`正在运行 Lumerical（已等待 ${Math.floor((performance.now()-t0)/1000)} s）…`; },500);
  try{
    const res=await fetch('/api/recognize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({width:300,height:300,pixels:snapshot,mode})});
    const data=await res.json(); if(!res.ok||data.error) throw new Error(data.error||('HTTP '+res.status));
    if(seq!==requestSeq){ return; }
    renderMain(data); statusEl.textContent = data.mode==='live' ? `实际 Lumerical 完成，${Number(data.elapsed_seconds).toFixed(1)} s。` : (data.mode==='cached'?'已保存光路矩阵计算完成（本次未启动求解器）。':'仅软件计算完成。');
  }catch(err){ if(seq===requestSeq){ clearResult(); statusEl.textContent='识别失败：'+err.message; detailsMain.textContent=String(err); } }
  finally{ btn.disabled=false; if(seq===requestSeq) busy=false; if(waitTimer){clearInterval(waitTimer);waitTimer=null;} }
};
document.getElementById('historyBtn').onclick=async()=>{
  try{
    const res=await fetch('/api/history'); const data=await res.json(); if(!res.ok||data.error) throw new Error(data.error||('HTTP '+res.status));
    document.getElementById('liveView').hidden=true; document.getElementById('liveBottom').hidden=true; document.getElementById('historyView').hidden=false;
    sourceBadge.textContent='历史回放 · 已完成的 Lumerical 仿真'; sourceBadge.className='cached';
    document.getElementById('historyRecord').textContent='记录：'+(data.record_id||'未记录');
    document.getElementById('historySource').textContent='历史回放 · 已完成的 Lumerical 仿真（约 '+(data.elapsed_seconds?Number(data.elapsed_seconds).toFixed(1):'未记录')+' s）';
    drawImage28('historyImage',data.image_28,1);
    const o=data.optical;
    drawChart('historyChart',data.digital.linear,o?o.restored_linear.real:null);
    document.getElementById('historyError').textContent=o?`16 路矩阵计算相对误差（历史记录）：${Number(o.linear_relative_error).toExponential(3)}`:'未记录光路对照';
    document.getElementById('historyPrediction').textContent=o?o.prediction:data.digital.prediction;
    document.getElementById('historyDigital').textContent=data.digital.prediction;
    document.getElementById('historyOptical').textContent=o?o.prediction:'未记录';
    document.getElementById('historyAgreement').textContent=o?(data.prediction_agreement?'当时软件与光路预测一致':'当时软件与光路预测不一致'):'历史记录缺光路结果';
    document.getElementById('historyDetail').innerHTML=detailHtml(data);
  }catch(err){ statusEl.textContent='历史记录读取失败：'+err.message; }
};
document.getElementById('back').onclick=()=>{ document.getElementById('historyView').hidden=true; document.getElementById('liveView').hidden=false; document.getElementById('liveBottom').hidden=false; if(lastResult) renderMain(lastResult); else clearResult(); };
fetch('/api/topology').then(r=>r.json()).then(topo=>{
  topology=topo;
  renderTopology('topoSvg',topo,true);
  renderTopology('historyTopoSvg',topo,false);
  selectedGate=placeMesh(topo.input_mesh.ops,90,410)[0];
  renderMzi(selectedGate);
});
document.getElementById('expandMzi').onclick=()=>{ if(topology){ selectedGate=placeMesh(topology.input_mesh.ops,90,410)[0]; renderMzi(selectedGate); document.getElementById('mziSvg').scrollIntoView({behavior:'smooth',block:'nearest'}); } };
function drawAutoDigit(){
  white(); hasInk=true; ctx.strokeStyle='#111'; ctx.lineWidth=20; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(150,55); ctx.lineTo(150,245); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(110,245); ctx.lineTo(190,245); ctx.stroke();
  clearResult();
}
const auto=new URLSearchParams(location.search).get('auto');
if(auto==='history') setTimeout(()=>document.getElementById('historyBtn').click(),250);
if(auto==='cached') setTimeout(()=>{ drawAutoDigit(); modeTouched=true; const r=document.querySelector('input[name="mode"][value="cached"]'); if(r)r.checked=true; document.getElementById('run').click(); },250);
fetch('/health').then(r=>r.json()).then(d=>{
  if(d.default_mode && !modeTouched && !auto){ const el=document.querySelector(`input[name="mode"][value="${d.default_mode}"]`); if(el) el.checked=true; }
  if(d.digital_only){ document.querySelectorAll('input[name="mode"]').forEach(el=>{el.disabled=true; if(el.value==='digital')el.checked=true;}); statusEl.textContent='服务以 --digital-only 启动：只能做软件基线。'; }
});
</script>
</body></html>"""


def _history_payload():
    if not HISTORY_RECORD.is_file():
        raise FileNotFoundError(f"missing history record: {HISTORY_RECORD}")
    data = json.loads(HISTORY_RECORD.read_text(encoding="utf-8"))
    data["historical"] = True
    data["record_id"] = HISTORY_RECORD.stem
    data["source"] = "history_saved_live"
    return data


def make_handler(args, lock):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            print(f"{self.address_string()} - {fmt % values}", flush=True)

        def _send(self, code, payload, content_type="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(
                payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route in ("/", "/index.html"):
                self._send(200, HTML.encode("utf-8"), "text/html")
            elif route == "/health":
                self._send(200, {"ok": True, "digital_only": args.digital_only,
                                 "default_mode": ("digital" if args.digital_only
                                                  else args.default_mode)})
            elif route == "/api/history":
                try:
                    self._send(200, _history_payload())
                except Exception as exc:
                    self._send(404, {"error": str(exc)})
            elif route == "/api/topology":
                try:
                    self._send(200, json.loads(TOPOLOGY_RECORD.read_text(encoding="utf-8")))
                except Exception as exc:
                    self._send(404, {"error": str(exc)})
            elif route.startswith("/api/session"):
                query = parse_qs(urlparse(self.path).query)
                try:
                    if query.get("record", [""])[0] == "history":
                        path = HISTORY_RECORD
                    else:
                        raw = query.get("path", [""])[0]
                        root = Path(args.out).resolve()
                        path = Path(raw).resolve()
                        if root != path and root not in path.parents:
                            raise ValueError("session path outside output directory")
                    if not path.is_file():
                        raise FileNotFoundError("session not found")
                    ctype = "application/json" if path.suffix == ".json" else "text/plain"
                    self._send(200, path.read_bytes(), ctype)
                except Exception as exc:
                    self._send(403, {"error": str(exc)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/recognize":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 8_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                width = int(payload["width"])
                height = int(payload["height"])
                pixels = payload["pixels"]
                mode = str(payload.get("mode", args.default_mode))
                if mode not in ("live", "cached", "digital"):
                    raise ValueError("mode must be live, cached or digital")
                if args.digital_only:
                    mode = "digital"
                with lock:
                    result = recognize_pixels(pixels, width, height,
                                              seed=args.seed, mode=mode,
                                              outdir=args.out)
                self._send(200, result)
            except Exception as exc:
                self._send(400, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--default-mode", choices=["cached", "live", "digital"],
                        default="live")
    parser.add_argument("--digital-only", action="store_true",
                        help="disable INTERCONNECT; label results as digital only")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args, lock))
    url = f"http://{args.host}:{args.port}/"
    print(f"Live digit UI: {url}", flush=True)
    print(f"sessions: {args.out}", flush=True)
    print(f"default mode: {'digital' if args.digital_only else args.default_mode}", flush=True)
    print(f"history: {HISTORY_RECORD}", flush=True)
    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
