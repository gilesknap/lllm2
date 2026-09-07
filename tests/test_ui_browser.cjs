// Run: node tests/test_ui_browser.cjs (CHROME_BIN can override the Chrome path).
// Uses only local HTML and mocked fetch; no panel, model, downloads or saved state.
// Chrome runs with an isolated profile. Screenshots remain in the printed temp directory.
function browser(){
const {spawn}=require('node:child_process');
const fs=require('node:fs');
const chrome=spawn(process.env.CHROME_BIN||'/opt/google/chrome/chrome',['--headless','--no-sandbox','--disable-gpu','--no-first-run','--disable-background-networking','--disable-component-update','--disable-sync','--remote-debugging-pipe','--user-data-dir='+artifacts+'/chrome'],{stdio:['ignore','ignore','inherit','pipe','pipe']});
chrome.on('exit',code=>{for(const p of pending.values())p.reject(Error('Chrome exited '+code));});
let seq=0,buffer='',pending=new Map();
chrome.stdio[4].on('data',chunk=>{buffer+=chunk;let end;while((end=buffer.indexOf('\0'))>=0){const msg=JSON.parse(buffer.slice(0,end));buffer=buffer.slice(end+1);if(msg.id){const p=pending.get(msg.id);pending.delete(msg.id);msg.error?p.reject(Error(JSON.stringify(msg.error))):p.resolve(msg.result);}}});
function send(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});chrome.stdio[3].write(JSON.stringify({id,method,params,sessionId})+'\0');});}
async function open(url,width=1440,height=900){const {targetId}=await send('Target.createTarget',{url:'about:blank'});const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});const call=(method,params)=>send(method,params,sessionId);await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});await call('Page.navigate',{url});return {call,eval:async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;},shot:async path=>{const r=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path,Buffer.from(r.data,'base64'));}};}
return {open,close:()=>chrome.kill()};

}

const fs=require('node:fs');
const path=require('node:path'),os=require('node:os');
const root=path.resolve(__dirname,'..'),artifacts=fs.mkdtempSync(path.join(os.tmpdir(),'lllm2-ui-'));
const b=browser();
const source=fs.readFileSync(root+'/lllm2/static/index.html','utf8').replace('<link rel="stylesheet" href="/static/panel.css">',()=>'<style>'+fs.readFileSync(root+'/lllm2/static/panel.css','utf8')+'</style>').replace('<script src="/static/panel.js"></script>',()=>'<script>'+fs.readFileSync(path.join(__dirname,'ui_fixture.js'),'utf8')+'</script><script>'+fs.readFileSync(root+'/lllm2/static/panel.js','utf8')+'</script>');
fs.writeFileSync(artifacts+'/after.html',source);
const assert=require('node:assert/strict');
(async()=>{try{
 const p=await b.open('file://'+artifacts+'/after.html');await new Promise(r=>setTimeout(r,600));
 const run=expr=>p.eval(expr.includes('await ')?'(async()=>{'+expr+'})()':expr);
 assert.equal(await run('$(' + JSON.stringify('start') + ').disabled'),false);
 await run("$('customize-toggle').click()");assert.equal(await run("$('customize').hidden"),false);
 await run("$('save-default').click(); new Promise(r=>setTimeout(r,40))");assert.equal(await run("$('customize').hidden"),false);
 assert.match(await run("$('settings-status').textContent"),/Saved/);
 await run("$('gpu-placement').value='manual';$('gpu-placement').dispatchEvent(new Event('change'));$('gpu_layers').value='0';$('gpu_layers').dispatchEvent(new Event('input'));new Promise(r=>setTimeout(r,300))");
 assert.equal(await run('settings().gpu_layers'),0);
 await run("$('save-default').click();new Promise(r=>setTimeout(r,40))");
 await run("loadDefaults('built-in')");assert.equal(await run('settings().gpu_layers'),null);
 await run("loadDefaults('saved')");assert.equal(await run('settings().gpu_layers'),0);
 await run("fixture.results=[{id:'r1',label:'Measured run',status:'complete',started:'2026-09-07',settings:{...fixture.settings,context:65536},samples:[{workload:'generate'}],probes:[]}];$('load-experiment').click();new Promise(r=>setTimeout(r,40))");
 assert.equal(await run("$('experiment-picker').open"),true);
 await run("$('experiment-options').querySelector('button').click();new Promise(r=>setTimeout(r,40))");assert.equal(await run('settings().context'),65536);
 assert.equal(await run('fixture.saved.context'),32768);
 await run("switchView('experiments')");await run("$('context').value=8192;edited('context');switchView('launch')");assert.equal(await run('settings().context'),65536);
 await run("switchView('experiments')");assert.equal(await run('settings().context'),8192);await run("switchView('launch')");
 assert.equal(await run("$('gpu-placement').value"),'auto');

 // An intervening poll must not re-enable a download whose POST is pending.
 await run("setModelFilter('catalog');fixture.delay=100;$('catalog-download-moe').click();await poll();$('catalog-download-moe').click();await new Promise(r=>setTimeout(r,200));fixture.delay=0");
 assert.equal(await run("fixture.posts.filter(p=>p.path==='/api/download').length"),1);
 // A slow rescan can finish after navigation or an edit without replacing the draft.
 await run("fixture.delay=100;scan();switchView('experiments');await new Promise(r=>setTimeout(r,400));fixture.delay=0");
 assert.equal(await run('view'),'experiments');await run("switchView('launch')");
 // Reordered settings responses must not overwrite an edit.
 await run("fixture.delay=100;loadDefaults('built-in');$('context').value=49152;edited('context');new Promise(r=>setTimeout(r,500))");
 assert.equal(await run('settings().context'),49152);await run('fixture.delay=0');
 await run("fixture.downloads=[{id:'dense',name:'Qwen3.8-27B',state:'downloading',percent:40,done_gb:6,total_gb:15.36,rate_mib_s:23,target:'/models/new/dense.gguf',detail:'Downloading'}];poll()");
 await run("setModelFilter('catalog');$('download-action-dense').focus();fixture.downloads[0].percent=50;poll()");
 assert.equal(await run('document.activeElement.id'),'download-action-dense');
 await run("setModelFilter('installed')");assert.equal(await run("$('download-section').hidden"),false);
 await run("fixture.models.push({path:'/models/new/dense.gguf',catalog_id:'dense',identity_verified:true,metadata:{context:262144}});fixture.downloads[0].state='complete';poll()");
 assert.equal(await run('settings().context'),49152);assert.equal(await run('settings().model'),'/models/Qwen3-8B/model.gguf');
 assert.equal(await run("$('download-action-dense').textContent"),'Use this model');
 // Ready guidance must use the running snapshot after edits.
 await run("fixture.engine={running:true,ready:true,pid:123,settings:{...fixture.settings,context:32768}};poll()");
 assert.match(await run("$('connect-context').textContent"),/32,768/);
 assert.match(await run("$('start').textContent"),/Restart/);
 assert.match(await run("$('ram').textContent"),/18.0 \/ 62.0 GiB/);
 await run("document.querySelector('[data-agent=claude]').click();Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async()=>{throw Error('Denied')}}});$('copy-agent').click();new Promise(r=>setTimeout(r,30))");
 assert.equal(await run("$('copy-dialog').open"),true);assert.equal(await run("$('copy-text').value"),'lllm2 claude');await run("$('copy-close').click()");
 await run("fixture.disconnected=true;poll()");assert.equal(await run("$('start').disabled"),true);assert.equal(await run("$('connect-agent').hidden"),true);assert.match(await run("$('resources-scope').textContent"),/last known/);
 await run("fixture.disconnected=false;poll()");assert.equal(await run("$('connect-agent').hidden"),false);
 await run("fixture.engine={running:false,ready:false,error:'CUDA out of memory (exact diagnostic)'};poll()");assert.match(await run("$('launch-error').textContent"),/Not enough memory/);assert.match(await run("$('error-raw').textContent"),/exact diagnostic/);
 assert.equal(await run("new Set([...document.querySelectorAll('[id]')].map(e=>e.id)).size===document.querySelectorAll('[id]').length"),true);
 // Render each representative state in both themes at desktop/mobile and 200% reflow.
 for(const state of ['idle','downloading','ready','edited','failed','fresh']){
  await run("loadedDefaults={settings:fixture.settings,mode:'recommended',source:'Estimated starting settings'};savedExists=false;$('message').style.display='none';fixture.engine={running:false,ready:false};fixture.downloads=[];fixture.checkError='';fixture.models=fixture.models.filter(m=>m.path===fixture.settings.model);fixture.engines=[{path:fixture.settings.engine,devices:['CUDA0']}];fill(fixture.settings);await inspect();customize(false);setModelFilter('installed');");
  if(state==='downloading')await run("fixture.downloads=[{id:'dense',name:'Qwen3.8-27B',state:'downloading',percent:40,done_gb:6,total_gb:15.36,rate_mib_s:23,target:'/models/new/dense.gguf',detail:'Downloading'}]");
  if(['ready','edited'].includes(state))await run("fixture.engine={running:true,ready:true,pid:123,settings:{...fixture.settings}};");
  if(state==='edited')await run("$('context').value=65536;edited('context');await inspect()");
  if(state==='failed')await run("fixture.engine.error='CUDA out of memory (exact diagnostic)'");
  if(state==='fresh')await run("fixture.models=[];fixture.engines=[];fill({...fixture.settings,model:'',engine:''});await scan();setModelFilter('catalog')");
  await run('poll()');
  await run('window.scrollTo(0,0)');
  for(const theme of ['light','dark'])for(const width of [1440,390,720]){
   await p.call('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:theme}]});
   await p.call('Emulation.setDeviceMetricsOverride',{width,height:width===390?844:900,deviceScaleFactor:width===720?2:1,mobile:width===390});
   assert.equal(await run('document.documentElement.scrollWidth<=innerWidth'),true,`overflow ${state}/${theme}/${width}`);
   await p.shot(`${artifacts}/${state}-${theme}-${width}.png`);
  }
 }
 // A completion on a fresh install must still wait for deliberate selection.
 await run("fixture.downloads=[{id:'dense',name:'Dense',state:'complete',percent:100,total_gb:15,done_gb:15}];fixture.models=[{path:'/models/new/dense.gguf',catalog_id:'dense',identity_verified:true,metadata:{context:262144}}];await poll()");
 assert.equal(await run('settings().model'),'');

 assert.equal(await run("$('settings-editor').contains($('feature-availability'))"),false);
 assert.equal(await run("$('feature-availability').parentElement.id"),'feature-host-launch');
 await run("switchView('experiments')");
 assert.equal(await run("$('feature-availability').parentElement.id"),'feature-host-experiments');
 await run("switchView('launch')");
 assert.equal(await run("getComputedStyle($('feature-availability')).borderTopStyle"),'solid');
 assert.equal(await run("getComputedStyle($('manage-models')).borderTopStyle"),'solid');
 // Actual keyboard activation, focus retention, and touch help.
 await run("fixture.models=[{path:fixture.settings.model,catalog_id:'qwen3-8b',metadata:{context:131072}}];fixture.engines=[{path:fixture.settings.engine,devices:['CUDA0']}];fixture.downloads=[];await scan();await selectModel(fixture.settings.model);customize(false);$('customize-toggle').focus()");
 await p.call('Page.bringToFront');await p.call('Emulation.setFocusEmulationEnabled',{enabled:true});
 const key=async key=>{await p.call('Input.dispatchKeyEvent',{type:'keyDown',key,code:key,text:key==='Enter'?'\r':'',windowsVirtualKeyCode:key==='Enter'?13:key==='Tab'?9:27});await p.call('Input.dispatchKeyEvent',{type:'keyUp',key,code:key,windowsVirtualKeyCode:key==='Enter'?13:key==='Tab'?9:27});};
 await key('Enter');assert.equal(await run("$('customize').hidden"),false);
 await run("$('load-menu').querySelector('summary').focus()");await key('Enter');assert.equal(await run("$('load-menu').open"),true);
 await key('Tab');assert.equal(await run('document.activeElement.id'),'built-in-default');await key('Escape');assert.equal(await run("$('load-menu').open"),false);
 const tip=await run("$('launch-tip-context').parentElement.querySelector('button').scrollIntoView({block:'center'});JSON.stringify((()=>{const r=$('launch-tip-context').parentElement.querySelector('button').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})())");
 await p.call('Emulation.setTouchEmulationEnabled',{enabled:true});await p.call('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[JSON.parse(tip)]});await p.call('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
 assert.equal(await run("$('launch-tip-context').parentElement.classList.contains('open')"),true);await key('Escape');
 await run("$('slot-note').scrollIntoView({block:'center'})");await p.shot(artifacts+'/customize-dividers.png');
 // Two rapid clicks still send only one launch request.
 await run("customize(false);fixture.engine={running:false,ready:false};await poll();$('start').click();$('start').click();await new Promise(r=>setTimeout(r,50))");
 assert.equal(await run("fixture.posts.filter(p=>p.path==='/api/start').length"),1);
 assert.equal(await run("$('start').textContent"),'Running');
 await run("renderHardware({gpus:[{name:'GPU',used_mib:null,total_mib:8192}],ram:{used_gib:null,total_gib:null,available_gib:null}})");
 assert.match(await run("$('vram').textContent"),/VRAM —/);assert.match(await run("$('ram').textContent"),/RAM — \/ —/);
 // Mixed result modes, numeric sorting, expandable evidence and spreadsheet exports.
 const baselineSettings=await run('fixture.settings');
 const resultFixture=(()=>{
  const base={settings:{...baselineSettings},probes:[],quality_status:'passed',status:'complete'};
  const sample={status:'complete',workload:'generate',input_tokens:1024,output_tokens:32,prefill_tok_s:120,decode_tok_s:25,peak_total_gpu_used_mib:4096,peak_engine_rss_mib:500};
  return [
   {...base,id:'cold',label:'Cold sample',started:'2026-09-07T09:02:00Z',samples:[sample],largest_observed_context:8192,recommended_context:7168},
   {...base,id:'old',label:'Zero and missing',started:'2026-09-07T09:01:00Z',samples:[{...sample,decode_tok_s:0},{status:'failed',workload:'edit',error:'Sample failed, with "details"',output_tokens:0}]},
   {...base,id:'warm',label:'Warm run',started:'2026-09-07T09:03:00Z',measurement_mode:'warm-conversation',samples:[{status:'partial',workload:'long-code',turn:'append',processed_prefill_tok_s:90,decode_tok_s:40,processed_tokens:20,reused_tokens:80,input_tokens:100,output_tokens:10},{status:'complete',control:true,turn:'append',processed_prefill_tok_s:400,decode_tok_s:100,reused_tokens:0}]},
   {...base,id:'failed',label:'No samples',started:'2026-09-07T09:04:00Z',status:'failed',samples:[],error:'Engine did not start'},
   {...base,id:'bad-source',label:'=SUM(1,2)\n"quoted"',started:'2026-09-07T09:03:30Z',quality_status:'failed',samples:[{...sample,decode_tok_s:999,adherence:{status:'failed'}}]}
  ];
 })();
 await run('fixture.results='+JSON.stringify(resultFixture));
 const savedBefore=await run('JSON.stringify(fixture.saved)');
 const mutationsBefore=await run("fixture.posts.filter(p=>['/api/start','/api/benchmark','/api/default/save'].includes(p.path)).length");
 await run("switchView('experiments')");
 assert.equal(await run("document.querySelectorAll('#results>tr[data-result-key]').length"),7);
 assert.equal(await run("document.querySelectorAll('#results>.result-detail:not([hidden])').length"),0);
 assert.equal(await run("$('detail-warm-0').querySelector('[data-promote]')"),null);
 assert.equal(await run("$('detail-bad-source-0').querySelector('[data-promote]')"),null);
 assert.equal(await run("$('detail-failed-0').querySelector('[data-promote]')"),null);
 await run("$('expand-cold-0').click();$('result-evidence-cold-0').open=true;$('expand-cold-0').focus();fixture.results[0].samples[0].decode_tok_s=75;await refreshResults()");
 assert.equal(await run("$('detail-cold-0').hidden"),false);
 assert.equal(await run("$('result-evidence-cold-0').open"),true);
 assert.equal(await run('document.activeElement.id'),'expand-cold-0');
 assert.match(await run("document.querySelector('[data-result-key=\"cold-0\"]').textContent"),/75/);
 await run("document.querySelector('[data-sort=decode]').click()");
 assert.equal(await run("document.querySelector('#results>tr[data-result-key]').dataset.resultKey"),'bad-source-0');
 await run("document.querySelector('[data-sort=decode]').click()");
 assert.equal(await run("document.querySelector('#results>tr[data-result-key]').dataset.resultKey"),'old-0');
 assert.deepEqual(await run("[...document.querySelectorAll('#results>tr[data-result-key]')].slice(-2).map(r=>r.dataset.resultKey)"),['old-1','failed-0']);
 await run("fixture.exports=[];downloadText=(text,type,name)=>fixture.exports.push({text,type,name});$('export-csv').click()");
 const exported=await run('fixture.exports[0]');assert.equal(exported.name,'lllm2-results.csv');
 const parsed=JSON.parse(require('node:child_process').execFileSync('python3',['-c','import csv,json,sys; print(json.dumps(list(csv.DictReader(sys.stdin))))'],{input:exported.text.replace(/^\ufeff/,''),encoding:'utf8'}));
 assert.equal(parsed.length,7);assert.equal(parsed[0].decode_tok_s,'0');assert.equal(parsed.find(r=>r.result_id==='failed').decode_tok_s,'');
 assert.equal(parsed.find(r=>r.result_id==='bad-source').label,"'=SUM(1,2)\n\"quoted\"");
 const warmRows=parsed.filter(r=>r.result_id==='warm');assert.equal(warmRows[0].prefill_tok_s,'');assert.equal(warmRows[0].processed_prefill_tok_s,'90');assert.equal(warmRows[1].sample_kind,'Uncached replay');
 assert.equal(parsed.find(r=>r.result_id==='cold').headroom_estimate_per_slot,'7168');
 await run("Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{fixture.copied=text}}});$('copy-results').click();await new Promise(r=>setTimeout(r,10))");
 assert.equal(await run('fixture.copied'),await run("tableText(resultRows(),'\t')"));
 await run("fixture.fullResults=[{id:'raw',samples:[{output:'Full evidence is retained'}]}];$('export').click();await new Promise(r=>setTimeout(r,20))");
 assert.equal(JSON.parse(await run('fixture.exports[1].text'))[0].samples[0].output,'Full evidence is retained');
 await run("Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async()=>{throw Error('Unavailable')}}});$('copy-results').click();await new Promise(r=>setTimeout(r,10))");
 assert.equal(await run("$('copy-text').value"),await run("tableText(resultRows(),'\t')"));assert.equal(await run("$('copy-text').tagName"),'TEXTAREA');await run("$('copy-close').click()");
 // The skip link must keep the active view and move keyboard focus into it.
 await run("$('skip-content').focus()");await key('Enter');assert.equal(await run('view'),'experiments');assert.equal(await run('document.activeElement.id'),'experiments-view');
 await run("switchView('launch');$('skip-content').focus()");await key('Enter');assert.equal(await run('document.activeElement.id'),'launch-view');
 await run("switchView('experiments')");assert.equal(await run("$('detail-cold-0').hidden"),false);
 assert.equal(await run('JSON.stringify(fixture.saved)'),savedBefore);
 assert.equal(await run("fixture.posts.filter(p=>['/api/start','/api/benchmark','/api/default/save'].includes(p.path)).length"),mutationsBefore);
 for(const theme of ['light','dark'])for(const width of [1440,390]){
  await p.call('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:theme}]});
  await p.call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:width===390});
  await run("$('comparisons').scrollIntoView({block:'start'})");
  assert.equal(await run('document.documentElement.scrollWidth<=innerWidth'),true);
  await p.shot(`${artifacts}/results-${theme}-${width}.png`);
 }
 console.log('Results checks passed: sorting, zero/missing metrics, modes, expansion/focus across refresh, eligibility, CSV quoting, multiline clipboard fallback, full JSON, skip links and unchanged drafts/defaults.');
 console.log('Artifacts: '+artifacts);
 console.log('Browser checks passed: toolbar, settings recovery, zero/Auto, separate drafts, stale response, download focus/completion, running context, clipboard fallback, disconnect, errors, unique IDs, 36 rendered state/theme/viewport combinations.');
}finally{b.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
setTimeout(()=>{b.close();process.exit(2);},60000).unref();
