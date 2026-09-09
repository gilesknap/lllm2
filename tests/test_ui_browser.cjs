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
const source=fs.readFileSync(root+'/src/lllm2/static/index.html','utf8').replace('<link rel="stylesheet" href="/static/panel.css">',()=>'<style>'+fs.readFileSync(root+'/src/lllm2/static/panel.css','utf8')+'</style>').replace('<script src="/static/panel.js"></script>',()=>'<script>'+fs.readFileSync(path.join(__dirname,'ui_fixture.js'),'utf8')+'</script><script>'+fs.readFileSync(root+'/src/lllm2/static/panel.js','utf8')+'</script>');
fs.writeFileSync(artifacts+'/after.html',source);
const assert=require('node:assert/strict');
(async()=>{try{
 const p=await b.open('file://'+artifacts+'/after.html');await new Promise(r=>setTimeout(r,600));
 assert.equal(await p.eval("document.getElementById('app-version').textContent"),'Version 1.2.3-test');
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

 // Find models sorts and filters metadata without changing launch/experiment drafts.
 await run("fixture.findEntries=[{id:'hf-a',name:'Small',display_name:'Small',repo:'test/Small-Instruct',file:'small-Q4_K_M.gguf',quant:'Q4_K_M',size_gb:2,fit:'Likely GPU fit',fit_rank:0,reason:'3 GiB reserved',task:'Chat / instruct',instruct:true,downloads:100,likes:2,updated:'2026-09-01',license:'apache-2.0'},{id:'hf-b',name:'Large',display_name:'Large',repo:'test/Large-Instruct',file:'large-Q4_K_M.gguf',quant:'Q4_K_M',size_gb:20,fit:'Likely needs CPU offload',fit_rank:1,reason:'4 GiB reserved',task:'Coding',instruct:true,downloads:200,likes:4,updated:'2026-09-02',license:'mit'}];await switchView('find')");
 assert.equal(await run("$('find-view').hidden"),false);
 assert.equal(await run("$('launch-view').hidden"),true);
 assert.equal(await run("$('find-table').querySelectorAll('tbody tr').length"),2);
 await run("$('find-table').querySelector('[data-find-sort=size_gb]').click();$('find-table').querySelector('[data-find-sort=size_gb]').click()");
 assert.match(await run("$('find-table').querySelector('tbody tr').textContent"),/Large/);
 await run("$('find-table').querySelector('[data-find-sort=size_gb]').click()");
 assert.equal(await run("$('find-table').querySelector('[data-find-sort=size_gb]').closest('th').getAttribute('aria-sort')"),'none');
 assert.match(await run("$('find-table').querySelector('tbody tr').textContent"),/Small/);
 // Include and exclude terms compose within a column and across columns.
 await run("const nameFilter=$('find-table').querySelector('[data-find-filter=display_name]');nameFilter.value='!sMaLl LAR';nameFilter.dispatchEvent(new Event('input',{bubbles:true}))");
 assert.deepEqual(await run("filteredFindEntries().map(e=>e.display_name)"),['Large']);
 await run("$('find-table').querySelector('[data-find-filter=license]').value='!mit';renderFind()");
 assert.equal(await run('filteredFindEntries().length'),0);
 await run("$('find-clear').click();$('find-table').querySelector('[data-find-filter=task]').value='!'+String.fromCharCode(34)+'chat / instruct'+String.fromCharCode(34);renderFind()");
 assert.deepEqual(await run("filteredFindEntries().map(e=>e.display_name)"),['Large']);
 await run("$('find-clear').click();$('find-table').querySelector('[data-find-filter=display_name]').value='!';renderFind()");
 assert.equal(await run('filteredFindEntries().length'),2);
 await run("$('find-clear').click()");
 // Many rows scroll inside the fixed viewport; titles and filters never overlap.
 await run("window.originalFindEntries=findEntries;findEntries=Array.from({length:40},(_,i)=>({...originalFindEntries[i%2],id:'row-'+i}));renderFind()");
 assert.equal(await run("$('find-results-scroll').clientHeight<=440&&$('find-results-scroll').scrollHeight>$('find-results-scroll').clientHeight"),true);
 await run("$('find-results-scroll').scrollTop=200;$('find-results-scroll').scrollLeft=900");
 assert.equal(await run("(()=>{const input=$('find-table').querySelector('[data-find-filter=task]'),r=input.getBoundingClientRect();return document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===input;})()"),true);
 await run("$('find-results-scroll').scrollTop=0;$('find-results-scroll').scrollLeft=0;findEntries=originalFindEntries;renderFind()");

 await run("const input=$('find-table').querySelector('[data-find-max=size_gb]');input.value='5';input.dispatchEvent(new Event('input',{bubbles:true}))");
 assert.equal(await run("$('find-table').querySelectorAll('tbody tr').length"),1);
 assert.match(await run("$('find-table').querySelector('tbody tr').textContent"),/Small/);
 await run("$('find-table').querySelector('[data-find-add]').click();new Promise(r=>setTimeout(r,40))");
 assert.equal(await run("fixture.catalog.some(e=>e.id==='hf-a')"),true);
 await run("$('catalog').querySelector('[data-catalogue-remove=hf-a]').click();new Promise(r=>setTimeout(r,40))");
 assert.equal(await run("$('remove-model-weights').checked"),false);
 await run("$('remove-model-cancel').click()");
 assert.equal(await run("fixture.catalog.some(e=>e.id==='hf-a')"),true);
 await run("$('catalog').querySelector('[data-catalogue-remove=hf-a]').click();await new Promise(r=>setTimeout(r,40));$('remove-model-weights').checked=true;$('remove-model-confirm').click();await new Promise(r=>setTimeout(r,100))");
 assert.equal(await run("fixture.posts.find(p=>p.path==='/api/catalogue/remove').data.delete_weights"),true);
 assert.equal(await run("fixture.catalog.some(e=>e.id==='hf-a')"),false);
 await run("$('find-clear').click()");
 for(const width of [1440,390]){
  await p.call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:width===390});
  assert.equal(await run('document.documentElement.scrollWidth<=innerWidth'),true);
  await p.shot(`${artifacts}/find-models-${width}.png`);
 }
 await p.call('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
 await run("switchView('launch')");
 assert.equal(await run('settings().context'),65536);
 await run("switchView('experiments')");assert.equal(await run('settings().context'),8192);await run("switchView('launch')");

 // An intervening poll must not re-enable a download whose POST is pending.
 await run("await switchView('find');fixture.delay=100;$('catalog-download-moe').click();await poll();$('catalog-download-moe').click();await new Promise(r=>setTimeout(r,200));fixture.delay=0");
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
 assert.equal(await run("$('download-action-dense').hidden"),true);await run("switchView('launch')");
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
   {...base,settings:{...baselineSettings,context:16384},id:'old',label:'Zero and missing',started:'2026-09-07T09:01:00Z',samples:[{...sample,decode_tok_s:0},{status:'failed',workload:'edit',error:'Sample failed, with "details"',output_tokens:0}]},
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
 for(const id of ['warm','bad-source','failed'])assert.equal(await run(`document.querySelector('#results [data-promote="${id}"]')`),null);
 // Launch actions remain visible on every eligible run's sample, without opening details.
 for(const key of ['cold-0','old-0','old-1'])assert.equal(await run(`$('promote-${key}').checkVisibility()`),true);
 assert.equal(await run("$('promote-cold-0').checkVisibility()"),true);
 assert.equal(await run("$('context-old-0')"),null);
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
 // After sorting, a later sample still loads its own run's settings, not another run's.
 await run("$('promote-old-1').click();await new Promise(r=>setTimeout(r,60))");
 assert.equal(await run('view'),'launch');assert.equal(await run('settings().context'),16384);
 assert.deepEqual(await run("fixture.posts.filter(p=>p.path==='/api/result/preview').at(-1).data"),{result_id:'old',use_context:false});
 await run("switchView('experiments');$('promote-cold-0').click();await new Promise(r=>setTimeout(r,60))");
 assert.equal(await run('settings().context'),8192);
 await run("switchView('experiments')");
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
 // Visible experiment controls: estimates react to inputs; selected settings stay independent.
 await run("fixture.engine={running:false,ready:false};await poll();await inspect();resetExperiments();$('clear-combos').click()");
 assert.equal(await run("$('combination-builder').checkVisibility() && !$('combination-builder').closest('details')"),true);
 assert.equal(await run("$('benchmark-time').checkVisibility()"),true);
 assert.deepEqual(await run("['search_context','sweep_prompts','full_window'].map(id=>$(id).checked)"),[true,false,false]);
 await run("$('sweep_prompts').click();$('search_context').click()");
 assert.match(await run("$('benchmark-time').textContent"),/2–3 minutes per configuration/);
 await run("$('repeats').value=2;$('repeats').dispatchEvent(new Event('input'))");
 assert.match(await run("$('benchmark-time').textContent"),/4–6 minutes/);
 await run("$('search_context').click()");
 assert.match(await run("$('benchmark-time').textContent"),/Context-search time is additional/);
 await run("$('full_window').click()");
 assert.match(await run("$('benchmark-time').textContent"),/no estimate/);
 await run("resetExperiments();document.querySelector('#workloads input[value=source-copy]').click()");
 assert.match(await run("$('benchmark-time').textContent"),/no estimate/);
 await run("document.querySelectorAll('#workloads input:checked').forEach(x=>x.click())");
 assert.match(await run("$('benchmark-time').textContent"),/Select a workload/);
 await run("resetExperiments();$('context').value=128;slotNote()");
 assert.match(await run("$('benchmark-time').textContent"),/Adjust the prompt/);
 await run("fill(fixture.settings);await inspect()");
 assert.equal(await run("document.querySelector('[data-mode=combinations]').disabled"),true);
 const benchBefore=await run("fixture.posts.filter(p=>p.path==='/api/benchmark').length");
 await run("$('add-combo').focus()");await key('Enter');
 await run("$('cache').value='q4_0';edited('cache');await inspect();$('add-combo').click()");
 assert.equal(await run("$('combo-count').textContent"),'2 selected');
 assert.deepEqual(await run('combinations.map(s=>s.cache)'),['q8_0','q4_0']);
 assert.equal(await run("fixture.posts.filter(p=>p.path==='/api/benchmark').length"),benchBefore);
 assert.equal(await run("document.querySelector('[data-mode=combinations]').disabled"),false);
 for(const theme of ['light','dark'])for(const width of [1440,390]){
  await p.call('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:theme}]});
  await p.call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:width===390});
  await run("$('benchmark-cost').closest('.benchmark-estimate').scrollIntoView({block:'start'});window.scrollBy(0,-90)");
  assert.equal(await run('document.documentElement.scrollWidth<=innerWidth'),true);
  await p.shot(`${artifacts}/experiment-controls-${theme}-${width}.png`);
 }
 await run("document.querySelector('[data-mode=combinations]').click();await new Promise(r=>setTimeout(r,50))");
 const submitted=await run("fixture.posts.filter(p=>p.path==='/api/benchmark').at(-1).data");
 assert.equal(submitted.mode,'combinations');assert.deepEqual(submitted.combinations.map(s=>s.cache),['q8_0','q4_0']);
 await run("$('clear-combos').click();await poll()");
 assert.equal(await run("$('combo-count').textContent"),'0 selected');
 assert.equal(await run("document.querySelector('[data-mode=combinations]').disabled"),true);
 assert.equal(await run('JSON.stringify(fixture.saved)'),savedBefore);
 // History deletion confirms whole runs, uses a fixed bulk selection, and keeps drafts/defaults.
 await run('fixture.results='+JSON.stringify(resultFixture));
 await run("fixture.results.push({...fixture.results[3],id:'cancelled',status:'cancelled'},{...fixture.results[3],id:'running',status:'running'});await refreshResults()");
 assert.equal(await run("$('comparisons').querySelector('h2').textContent"),'Experiment history');
 assert.equal(await run("$('delete-run-running-0').disabled"),true);
 assert.match(await run("$('delete-failed-results').textContent"),/\(2\)/);
 await run("$('delete-failed-results').click()");
 assert.equal(await run("$('delete-results-dialog').open"),true);
 assert.equal(await run("document.activeElement.id"),'delete-results-cancel');
 await run("$('delete-results-cancel').click()");
 assert.equal(await run("fixture.results.some(r=>r.id==='failed')"),true);
 await run("$('delete-failed-results').click();fixture.results.push({...fixture.results[3],id:'later-failure'});$('delete-results-confirm').click();await new Promise(r=>setTimeout(r,70))");
 assert.deepEqual(await run("fixture.posts.filter(p=>p.path==='/api/results/delete').at(-1).data"),{result_ids:['failed','cancelled'],failed_only:true});
 assert.equal(await run("fixture.results.some(r=>r.id==='later-failure')"),true);
 assert.equal(await run("fixture.results.some(r=>r.id==='bad-source')"),true);
 await run("fixture.job={status:'running',active:true};await poll()");
 assert.equal(await run("$('delete-failed-results').disabled"),true);
 assert.equal(await run("$('delete-run-old-1').disabled"),true);
 await run("fixture.job={status:'idle',active:false};await poll();await previewResult('old');await switchView('experiments')");
 const draftBeforeDelete=await run('JSON.stringify(settings())');
 await run("$('delete-run-old-1').click()");
 assert.match(await run("$('delete-results-description').textContent"),/2 samples/);
 await run("$('delete-results-confirm').click();await new Promise(r=>setTimeout(r,70))");
 assert.equal(await run("fixture.results.some(r=>r.id==='old')"),false);
 assert.equal(await run("document.querySelector('[data-result-key=old-0]')"),null);
 assert.equal(await run('JSON.stringify(settings())'),draftBeforeDelete);
 assert.equal(await run('JSON.stringify(fixture.saved)'),savedBefore);
 await run("switchView('launch')");
 assert.equal(await run('loadedDefaults.mode'),'custom');
 assert.equal(await run('settings().context'),16384);
 // Tested context loads the measured maximum unless the optional margin is selected.
 await run("switchView('experiments')");
 assert.equal(await run("document.querySelector('#results [data-context-choice=headroom]').checked"),false);
 assert.equal(await run("$('promote-cold-0').textContent"),'Try in Launch');
 assert.equal(await run("$('promote-cold-0').parentElement.querySelector('[data-context-choice=headroom]').checked"),false);
 await run("$('promote-cold-0').click();await new Promise(r=>setTimeout(r,60))");
 assert.equal(await run('settings().context'),8192);
 assert.deepEqual(await run("fixture.posts.filter(p=>p.path==='/api/result/preview').at(-1).data"),{result_id:'cold',use_context:true,reserve_headroom:false});
 await run("switchView('experiments');document.querySelector('#results [data-context-choice=headroom]').click();$('promote-cold-0').click();await new Promise(r=>setTimeout(r,60))");
 assert.equal(await run('settings().context'),7168);
 assert.equal(await run('loadedDefaults.reserve_headroom'),true);
 assert.equal(await run("contextChoice"),'headroom');
 await run("document.querySelector('#results [data-context-choice=headroom]').click();$('save-default').click();await new Promise(r=>setTimeout(r,60))");
 assert.deepEqual(await run("fixture.posts.filter(p=>p.path==='/api/default/save').at(-1).data"),{result_id:'cold',use_context:true,reserve_headroom:true});
 // Clearing both options restores the original context; picker shares the same choices.
 assert.equal(await run("document.querySelectorAll('[data-context-choice]:checked').length"),0);
 await run("switchView('experiments');$('promote-cold-0').click();await new Promise(r=>setTimeout(r,60))");
 assert.equal(await run('settings().context'),baselineSettings.context);
 assert.deepEqual(await run("fixture.posts.filter(p=>p.path==='/api/result/preview').at(-1).data"),{result_id:'cold',use_context:false});
 await run("$('load-experiment').click();await new Promise(r=>setTimeout(r,60))");
 assert.equal(await run("document.querySelectorAll('#experiment-options [data-result=cold]').length"),1);
 await run("document.querySelector('#experiment-options [data-context-choice=headroom]').click()");
 assert.equal(await run("document.querySelectorAll('[data-context-choice=tested]:checked').length"),0);
 await run("document.querySelector('#experiment-options [data-context-choice=tested]').click()");
 assert.equal(await run("document.querySelectorAll('[data-context-choice=headroom]:checked').length"),0);
 await run("document.querySelector('#experiment-options [data-result=cold]').click();await new Promise(r=>setTimeout(r,60))");
 assert.equal(await run('settings().context'),8192);
 console.log('Experiment controls passed: visible controls, reactive estimates, keyboard add, independent combinations, empty selection guard, mocked submission and responsive layouts.');
 console.log('Results checks passed: sorting, zero/missing metrics, modes, expansion/focus across refresh, eligibility, CSV quoting, multiline clipboard fallback, full JSON, skip links and unchanged drafts/defaults.');
 console.log('Artifacts: '+artifacts);
 console.log('Browser checks passed: toolbar, settings recovery, zero/Auto, separate drafts, stale response, download focus/completion, running context, clipboard fallback, disconnect, errors, unique IDs, 36 rendered state/theme/viewport combinations.');
}finally{b.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
setTimeout(()=>{b.close();process.exit(2);},60000).unref();
