#!/usr/bin/env node
// Build the static site into docs/ (served by GitHub Pages → Deploy from /docs).
//   node build.js
//
// Reads cheatsheets/*.md and drills/*/README.md, wraps each into _template.html
// (markdown embedded inline so each page is single-file & opens from file://),
// and writes a searchable, responsive index hub.

const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const OUT = path.join(ROOT, 'docs');
const tpl = fs.readFileSync(path.join(ROOT, '_template.html'), 'utf8');
fs.mkdirSync(OUT, { recursive: true });

function titleOf(md, fallback) {
  const m = md.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : fallback;
}
function renderDoc(md, title, outFile) {
  const safe = md.replace(/<\/script>/gi, '<\\/script>');
  const html = tpl.replace('{{TITLE}}', () => title).replace('{{MARKDOWN}}', () => safe);
  fs.writeFileSync(path.join(OUT, outFile), html);
}

const items = [];
for (const f of fs.readdirSync(path.join(ROOT, 'cheatsheets')).filter((f) => f.endsWith('.md')).sort()) {
  const slug = f.replace(/\.md$/, '');
  const md = fs.readFileSync(path.join(ROOT, 'cheatsheets', f), 'utf8');
  const out = 'cheatsheet-' + slug + '.html';
  renderDoc(md, titleOf(md, slug), out);
  items.push({ section: 'Cheatsheets 题解', title: titleOf(md, slug), href: out });
}
for (const d of fs.readdirSync(path.join(ROOT, 'drills')).sort()) {
  const rp = path.join(ROOT, 'drills', d, 'README.md');
  if (!fs.existsSync(rp)) continue;
  const md = fs.readFileSync(rp, 'utf8');
  const out = 'drill-' + d + '.html';
  renderDoc(md, titleOf(md, d), out);
  items.push({ section: 'Drills 手撕', title: titleOf(md, d), href: out });
}

function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }
function buildIndex(items) {
  const groups = {};
  for (const it of items) (groups[it.section] = groups[it.section] || []).push(it);
  const sections = Object.keys(groups).map((sec) => {
    const cards = groups[sec].map((it) =>
      `<a class="card" data-t="${esc((it.title + ' ' + it.href).toLowerCase())}" href="${esc(it.href)}">${esc(it.title)}</a>`
    ).join('');
    return `<section><h2>${esc(sec)} <small>${groups[sec].length}</small></h2><div class="grid">${cards}</div></section>`;
  }).join('\n');
  return `<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Post-Training Playbook</title>
<style>
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;font:16px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;color:#222;background:#fafafa}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #e6e6e6;padding:.9rem 1rem;z-index:5}
h1{margin:.1rem 0;font-size:1.15rem}
.sub{color:#888;font-size:.8rem;margin:.2rem 0 .6rem}
#q{width:100%;max-width:980px;padding:.6rem .8rem;font-size:1rem;border:1px solid #ccc;border-radius:10px;background:transparent;color:inherit}
main{max-width:980px;margin:0 auto;padding:1rem}
h2{font-size:1rem;margin:1.4rem 0 .6rem;color:#0a7}
h2 small{color:#999;font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:.6rem}
.card{display:block;padding:.7rem .8rem;border:1px solid #e6e6e6;border-radius:10px;background:#fff;text-decoration:none;color:inherit;font-size:.9rem;line-height:1.35}
.card:hover{border-color:#0a7;transform:translateY(-1px)}
footer{max-width:980px;margin:1rem auto 3rem;padding:0 1rem;color:#999;font-size:.78rem}
.hidden{display:none}
@media(prefers-color-scheme:dark){
 body{color:#ddd;background:#16181c}
 header{background:#1b1e23;border-color:#2c2f36}
 .card{background:#22262c;border-color:#2c2f36}
}
</style></head>
<body>
<header>
<h1>Post-Training Playbook</h1>
<div class="sub">${items.length} 篇 · LLM 后训练面试复习 · 公式 MathJax + 代码高亮 · 手机/电脑均可读 · 输入关键词实时过滤</div>
<input id="q" type="search" placeholder="搜索主题… / filter topics…" autocomplete="off" autofocus>
</header>
<main>${sections}</main>
<footer>AI 辅助整理的学习笔记,WIP,欢迎 issue/PR 纠错。由 <code>node build.js</code> 生成。</footer>
<script>
const q=document.getElementById('q');
q.addEventListener('input',function(){
  const v=q.value.trim().toLowerCase();
  document.querySelectorAll('.card').forEach(function(c){
    c.classList.toggle('hidden', v.length>0 && c.dataset.t.indexOf(v)<0);
  });
  document.querySelectorAll('main section').forEach(function(s){
    const any=Array.prototype.some.call(s.querySelectorAll('.card'),function(c){return !c.classList.contains('hidden');});
    s.classList.toggle('hidden', !any);
  });
});
</script>
</body></html>`;
}

fs.writeFileSync(path.join(OUT, 'index.html'), buildIndex(items));
console.log('Built ' + items.length + ' pages into docs/');
