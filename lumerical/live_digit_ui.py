"""Local canvas UI for one handwritten digit through the 16-channel layer.

Run with the project's normal Python (NumPy only); the server uses the standard
library HTTP stack.  When "live INTERCONNECT" is checked, each recognition
builds and runs the 16-channel time-domain circuit; it does not use H@x.
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

HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>16 通道光学手写数字</title>
<style>
:root { color-scheme: light; }
body { font-family: "Segoe UI", system-ui, sans-serif; margin: 0; background: #f4f6f8; color: #17202a; }
main { max-width: 980px; margin: 0 auto; padding: 22px; }
h1 { font-size: 23px; margin: 0 0 4px; }
.sub { color: #5b6770; margin-bottom: 18px; }
.grid { display: grid; grid-template-columns: 340px 1fr; gap: 22px; }
.card { background: #fff; border: 1px solid #d9e0e6; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px #00000008; }
canvas { width: 280px; height: 280px; border: 1px solid #b9c4cc; border-radius: 8px; touch-action: none; cursor: crosshair; }
.row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
button { border: 0; border-radius: 8px; padding: 9px 13px; font-size: 14px; cursor: pointer; background: #e9eef2; }
button.primary { background: #1769aa; color: #fff; }
button:disabled { opacity: .55; cursor: wait; }
label { font-size: 14px; color: #35424c; display: flex; gap: 7px; align-items: center; }
#status { margin-top: 12px; min-height: 22px; color: #5b6770; font-size: 14px; }
.big { font-size: 30px; font-weight: 700; letter-spacing: .06em; }
.muted { color: #66727c; font-size: 13px; }
.pre { font-family: Consolas, monospace; font-size: 12px; white-space: pre-wrap; background: #f7f9fa; border-radius: 8px; padding: 9px; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.badge { display: inline-block; border-radius: 999px; padding: 5px 10px; font-weight: 700; font-size: 12px; letter-spacing: .04em; background: #e8edf1; color: #45525c; }
.badge.live { background: #d93025; color: #fff; }
.badge.cached { background: #e9eef2; color: #45525c; }
.badge.digital { background: #e9eef2; color: #45525c; }
.proof { margin-top: 12px; border-top: 1px solid #e1e7eb; padding-top: 12px; }
.proofgrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0; }
.metric { background: #f7f9fa; border-radius: 8px; padding: 8px; }
.metric b { display: block; font-size: 16px; }
.metric span { color: #66727c; font-size: 11px; }
#chart { width: 100%; height: 190px; border: 1px solid #e1e7eb; border-radius: 8px; background: #fff; }
.legend { font-size: 12px; color: #5b6770; margin: 5px 0; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px; }
.pill { display: inline-block; background: #edf4fb; color: #1769aa; border-radius: 999px; padding: 3px 9px; font-size: 12px; }
.warn { color: #9a5b00; }
@media(max-width: 760px) { .grid { grid-template-columns: 1fr; } canvas { width: 260px; height: 260px; } }
</style>
</head>
<body><main>
<h1>16 通道光学手写数字</h1>
<div class="sub">画一个数字。数字基线始终计算。模式三选一：实际 INTERCONNECT 每张图都建立并运行 16 通道时域电路；缓存 H 明确标为演示；仅数字不调用求解器。</div>
<div class="grid">
  <section class="card">
    <canvas id="pad" width="280" height="280"></canvas>
    <div class="row">
      <button id="clear">清空</button>
      <button id="uploadBtn">上传图片</button>
      <input id="file" type="file" accept="image/*" hidden>
    </div>
    <div class="row">
      <label><input type="radio" name="mode" value="live" checked> 实际 INTERCONNECT</label>
      <label><input type="radio" name="mode" value="cached"> 缓存 H 演示</label>
      <label><input type="radio" name="mode" value="digital"> 仅数字</label>
      <button id="run" class="primary">识别</button>
    </div>
    <div id="status">准备就绪。</div>
  </section>
  <section class="card">
    <div class="two">
      <div><div class="muted">28×28 输入</div><canvas id="preview" width="280" height="280" style="width:140px;height:140px"></canvas></div>
      <div>
        <div class="muted">预测</div>
        <div class="big" id="pred">—</div>
        <div id="agreement" class="muted"></div>
        <div class="muted" style="margin-top:10px">输入特征 x</div>
        <div id="feat" class="pre">—</div>
      </div>
    </div>
    <div id="details" class="pre" style="margin-top:12px">尚未识别。</div>
    <div id="proof" class="proof">
      <span id="badge" class="badge">—</span>
      <div id="proofText" class="muted" style="margin-top:8px">识别后这里显示是否实际调用了 Lumerical、电路规模、耗时、编码和原始复输出。</div>
      <div id="proofGrid" class="proofgrid"></div>
      <div class="legend"><span class="sw" style="background:#1769aa"></span>数字线性输出 <span class="sw" style="background:#d97706"></span>光路恢复线性输出</div>
      <canvas id="chart" width="560" height="190"></canvas>
    </div>
  </section>
</div>
</main>
<script>
const pad = document.getElementById('pad'), ctx = pad.getContext('2d');
const preview = document.getElementById('preview'), pctx = preview.getContext('2d');
const statusEl = document.getElementById('status'), details = document.getElementById('details');
let drawing = false, last = null;
function white(){ ctx.fillStyle = '#fff'; ctx.fillRect(0,0,280,280); ctx.strokeStyle = '#111'; ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; }
white();
function pos(e){ const r = pad.getBoundingClientRect(); return {x:(e.clientX-r.left)*280/r.width, y:(e.clientY-r.top)*280/r.height}; }
pad.addEventListener('pointerdown', e => { drawing=true; last=pos(e); pad.setPointerCapture(e.pointerId); ctx.beginPath(); ctx.moveTo(last.x,last.y); });
pad.addEventListener('pointermove', e => { if(!drawing) return; const p=pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); last=p; });
pad.addEventListener('pointerup', () => { drawing=false; last=null; });
document.getElementById('clear').onclick = () => { white(); details.textContent='尚未识别。'; statusEl.textContent='已清空。'; };
const file = document.getElementById('file');
document.getElementById('uploadBtn').onclick = () => file.click();
file.onchange = () => { const f=file.files[0]; if(!f) return; const img=new Image(); img.onload=()=>{ white(); const s=Math.min(280/img.width,280/img.height); const w=img.width*s,h=img.height*s; ctx.fillStyle='#fff'; ctx.fillRect(0,0,280,280); ctx.drawImage(img,(280-w)/2,(280-h)/2,w,h); statusEl.textContent='图片已载入，可识别。'; }; img.src=URL.createObjectURL(f); };
function canvasPixels(){ const d=ctx.getImageData(0,0,280,280).data, out=[]; for(let i=0;i<d.length;i+=4) out.push((d[i]+d[i+1]+d[i+2])/3); return out; }
function showImage(img28){ pctx.fillStyle='#fff'; pctx.fillRect(0,0,280,280); for(let y=0;y<28;y++) for(let x=0;x<28;x++){ const v=255-img28[y][x]; pctx.fillStyle=`rgb(${v},${v},${v})`; pctx.fillRect(x*10,y*10,10,10); } }
function fmt(a){ return a.map(v => Number(v).toFixed(3)).join(', '); }
function drawChart(digital, optical){
  const c=document.getElementById('chart'), g=c.getContext('2d');
  g.clearRect(0,0,c.width,c.height); g.fillStyle='#fff'; g.fillRect(0,0,c.width,c.height);
  const n=digital.length, left=30, right=8, top=12, bottom=24;
  const w=(c.width-left-right)/n, mid=top+(c.height-top-bottom)/2;
  const max=Math.max(...digital.map(Math.abs), ...optical.map(Math.abs), 1e-9);
  const scale=(mid-top)/max;
  g.strokeStyle='#c8d1d8'; g.beginPath(); g.moveTo(left,mid); g.lineTo(c.width-right,mid); g.stroke();
  g.font='10px Consolas'; g.fillStyle='#66727c';
  for(let i=0;i<n;i++){
    const x=left+i*w+w*0.16;
    const bh=Math.abs(digital[i])*scale, oh=Math.abs(optical[i])*scale;
    g.fillStyle='#1769aa'; g.fillRect(x, digital[i]>=0?mid-bh:mid, w*0.30, bh);
    g.fillStyle='#d97706'; g.fillRect(x+w*0.34, optical[i]>=0?mid-oh:mid, w*0.30, oh);
    g.fillText(String(i+1), x, c.height-7);
  }
  g.fillStyle='#66727c'; g.fillText('0', 8, mid+3);
}
function showProof(data){
  const badge=document.getElementById('badge'), text=document.getElementById('proofText'), grid=document.getElementById('proofGrid');
  const optical=data.optical;
  if(!optical){
    badge.className='badge digital'; badge.textContent='DIGITAL';
    text.textContent='本次没有调用 Lumerical；只计算数字基线。';
    grid.innerHTML=''; drawChart(data.digital.linear, data.digital.linear);
    return;
  }
  const live=!!data.live_interconnect_called;
  badge.className='badge '+(live?'live':(data.mode==='cached'?'cached':'digital'));
  badge.textContent=live?'LIVE 16 通道':'CACHED H';
  const raw=optical.raw_optical;
  const sessionLink=data.session_file ? `<a href="/api/session?path=${encodeURIComponent(data.session_file)}">下载本次 session JSON</a>` : '';
  text.innerHTML=`<b>${live?'本次实际调用 INTERCONNECT':'缓存 H 演示，未调用求解器'}</b><br>`+
    `trace: ${(optical.run_trace||[]).map(x=>'<code>'+x+'</code>').join(' → ')||'—'}<br>`+
    `run_trace 是本次执行步骤；live 模式会真实 build + app.run + read_iq。 ${sessionLink}`;
  const metrics=[
    ['模式', live?'实际光路':(data.mode==='cached'?'缓存 H':'数字')],
    ['电路', live?`${optical.circuit_size} 通道 / ${optical.mzi_count} MZI`:`${optical.circuit_size} 通道（缓存）`],
    ['耗时', live?`build ${Number(optical.build_seconds).toFixed(1)} s + run ${Number(optical.run_seconds).toFixed(1)} s`:'未调用求解器'],
    ['线性误差', Number(optical.linear_relative_error).toExponential(2)],
    ['复泄漏', Number(optical.imaginary_leakage_ratio).toExponential(2)],
    ['g / s', `${Number(optical.encoding_scale_g).toFixed(4)} / ${Number(optical.recovery_scale_s).toFixed(4)}`],
    ['Lumerical', live?(optical.lumerical_version||'已调用').split('\n')[0]:'未调用'],
    ['预测', `${data.digital.prediction} / ${optical.prediction}`],
  ];
  grid.innerHTML=metrics.map(([k,v])=>`<div class="metric"><span>${k}</span><b>${v}</b></div>`).join('');
  const amp=optical.encoded_amplitude||[], ph=optical.encoded_phase_rad||[];
  const rawText=raw.real.map((v,i)=>`${String(i+1).padStart(2,'0')}  y=${v.toFixed(6)} ${raw.imag[i]>=0?'+':''}${raw.imag[i].toFixed(6)}j  |y|=${Math.hypot(v,raw.imag[i]).toFixed(6)}`).join('\n');
  document.getElementById('details').textContent =
    `encoded |x|/g: ${fmt(amp)}\nencoded phase: ${fmt(ph)}\nraw y:\n${rawText}\n\n`+
    JSON.stringify({status:data.status,mode:data.mode,live_interconnect_called:data.live_interconnect_called,elapsed_s:data.elapsed_seconds,linear_relative_error:optical.linear_relative_error,imaginary_leakage_ratio:optical.imaginary_leakage_ratio,digital_prediction:data.digital.prediction,optical_prediction:optical.prediction,session_file:data.session_file||null}, null, 2);
  drawChart(data.digital.linear, optical.restored_linear.real);
}
document.getElementById('run').onclick = async () => {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const btn = document.getElementById('run'); btn.disabled=true; statusEl.textContent = mode === 'live' ? '正在建立并运行 16 通道 INTERCONNECT（可能较慢）…' : (mode === 'cached' ? '正在应用缓存 H…' : '正在计算数字基线…');
  try {
    const res = await fetch('/api/recognize', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({width:280,height:280,pixels:canvasPixels(),mode})});
    const data = await res.json();
    if(!res.ok || data.error) throw new Error(data.error || ('HTTP '+res.status));
    showImage(data.image_28);
    document.getElementById('pred').textContent = `数字 ${data.digital.prediction}` + (data.optical ? ` / 光路 ${data.optical.prediction}` : '');
    document.getElementById('agreement').textContent = data.optical ? (data.prediction_agreement ? '数字与光路预测一致' : '数字与光路预测不一致') : '仅数字基线';
    document.getElementById('feat').textContent = fmt(data.features_x);
    details.textContent = '识别完成。';
    showProof(data);
    statusEl.textContent = data.live_interconnect_called ? '实际 INTERCONNECT 已完成。' : (data.mode === 'cached' ? '缓存 H 演示完成（非现场求解）。' : '数字基线完成（未调用求解器）。');
  } catch(e) { statusEl.textContent = '失败：' + e.message; details.textContent = String(e.stack || e); }
  finally { btn.disabled=false; }
};
fetch('/health').then(r=>r.json()).then(d=>{ if(d.digital_only){ document.querySelectorAll('input[name="mode"]').forEach(el=>{el.checked=(el.value==='digital'); el.disabled=true;}); statusEl.textContent='服务以 --digital-only 启动，不会调用 INTERCONNECT。'; }});
</script></body></html>"""


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
            if self.path in ("/", "/index.html"):
                self._send(200, HTML.encode("utf-8"), "text/html")
            elif self.path == "/health":
                self._send(200, {"ok": True, "digital_only": args.digital_only})
            elif self.path.startswith("/api/session"):
                query = parse_qs(urlparse(self.path).query)
                raw = query.get("path", [""])[0]
                try:
                    root = Path(args.out).resolve()
                    path = Path(raw).resolve()
                    if root != path and root not in path.parents:
                        raise ValueError("session path outside output directory")
                    if not path.is_file():
                        raise FileNotFoundError("session not found")
                    content_type = ("application/json" if path.suffix == ".json"
                                    else "text/plain")
                    self._send(200, path.read_bytes(), content_type)
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
                mode = str(payload.get("mode", "live"))
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
